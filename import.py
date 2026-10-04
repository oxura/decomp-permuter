#!/usr/bin/env python3
# usage: ./import.py path/to/file.c path/to/asm.s [make flags]
import argparse
from base64 import b64encode
from collections import defaultdict
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import toml
import fnmatch
from typing import (
    Callable,
    Dict,
    List,
    Match,
    Mapping,
    Optional,
    Pattern,
    Tuple,
    Set,
    cast,
)
import urllib.request
import urllib.parse

from src import ast_util
from src.compiler import Compiler
from src.error import CandidateConstructionFailure
from src.helpers import get_default_randomization_weights

is_macos = platform.system() == "Darwin"


def homebrew_gcc_cpp() -> str:
    lookup_paths = ["/usr/local/bin", "/opt/homebrew/bin"]

    for lookup_path in lookup_paths:
        try:
            return max(f for f in os.listdir(lookup_path) if f.startswith("cpp-"))
        except ValueError:
            pass

    print(
        "Error while looking up in " + ":".join(lookup_paths) + " for cpp- executable"
    )
    sys.exit(1)


cpp_cmd = homebrew_gcc_cpp() if is_macos else "cpp"
make_cmd = "gmake" if is_macos else "make"

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_ASM_PRELUDE_FILE = os.path.join(dir_path, "prelude.inc")

DEFAULT_AS_CMDLINE: List[str] = ["mips-linux-gnu-as", "-march=vr4300", "-mabi=32"]

RE_FUNC_NAME = r"[a-zA-Z0-9_$]+"

CPP: List[str] = [cpp_cmd, "-P", "-undef"]

STUB_FN_MACROS: List[str] = [
    "-D_Static_assert(x, y)=",
    "-D__attribute__(x)=",
    "-DGLOBAL_ASM(...)=",
    "-D__asm__(...)=_permuter_ignore_line __asm__(__VA_ARGS__)",
]

SETTINGS_FILES = [
    "permuter_settings.toml",
    "tools/permuter_settings.toml",
    "config/permuter_settings.toml",
]


def json_dict(
    data: Mapping[str, object], prop: str, allow_missing: bool = True
) -> Dict[str, object]:
    if prop not in data:
        if allow_missing:
            return {}
        else:
            raise Exception(f'missing "{prop}" property')
    ret = data[prop]
    if not isinstance(ret, dict):
        raise Exception(f'"{prop}" property must be a dict')
    return cast(Dict[str, object], ret)


def formatcmd(cmdline: List[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in cmdline)


def prune_asm(asm_cont: str) -> Tuple[str, str]:
    func_name = None
    asm_lines: List[str] = []
    late_rodata: List[str] = []
    cur_section = ".text"
    for line in asm_cont.splitlines(keepends=True):
        changed_section = False
        line_parts = line.split()

        if len(line_parts) >= 2 and line_parts[0] == ".section":
            cur_section = line_parts[1]
            changed_section = True
        elif line.strip() in [
            ".text",
            ".rdata",
            ".rodata",
            ".late_rodata",
            ".bss",
            ".data",
        ]:
            cur_section = line.strip()
            changed_section = True

        if cur_section == ".late_rodata":
            if not changed_section:
                late_rodata.append(line)
            continue

        if (
            func_name is None
            and cur_section == ".text"
            and len(line_parts) >= 2
            and line_parts[0] in ("glabel", ".globl")
        ):
            func_name = line_parts[1]
        asm_lines.append(line)

    # ".late_rodata" is non-standard asm, so we add it to the end of the file as ".rodata"
    if late_rodata:
        asm_lines.extend(["\n.section .rodata\n"] + late_rodata)

    if func_name is None:
        print(
            "Missing function name in assembly file! The file should start with 'glabel function_name'.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not re.fullmatch(RE_FUNC_NAME, func_name):
        print(f"Bad function name: {func_name}", file=sys.stderr)
        sys.exit(1)

    return func_name, "".join(asm_lines)


def find_global_asm_func(root_dir: str, c_file: str, func_name: str) -> str:
    try:
        with open(c_file) as f:
            source = f.read()
    except OSError as e:
        print("Could not open C file:", e, file=sys.stderr)
        sys.exit(1)

    global_asm: Optional[List[str]] = None
    for line in source.splitlines(keepends=True):
        stop_global_asm = False
        if global_asm is not None:
            if line.startswith(")"):
                stop_global_asm = True
            else:
                global_asm.append(line)
        else:
            if not line.startswith("GLOBAL_ASM(") and not line.startswith(
                "#pragma GLOBAL_ASM("
            ):
                continue
            global_asm = []
            if ")" in line:
                ind = line.index("(")
                ind2 = line.index(")", ind)
                fname = os.path.join(root_dir, line[ind + 2 : ind2 - 1])
                try:
                    with open(fname) as f2:
                        for line2 in f2:
                            global_asm.append(line2)
                except OSError as e:
                    print(
                        f"Failed to open GLOBAL_ASM file {fname}:", e, file=sys.stderr
                    )
                    sys.exit(1)
                stop_global_asm = True
        if stop_global_asm:
            for line2 in global_asm:
                if "glabel " + func_name in line2:
                    return "".join(global_asm)
            global_asm = None
    print(f"Failed to find GLOBAL_ASM for function {func_name}.", file=sys.stderr)
    sys.exit(1)


def parse_asm(
    root_dir: str, c_file: str, asm_file_or_func_name: str
) -> Tuple[str, str]:
    require_matching_func_name = False
    try:
        with open(asm_file_or_func_name, encoding="utf-8") as f:
            asm_cont = f.read()
    except OSError as e:
        if re.fullmatch(RE_FUNC_NAME, asm_file_or_func_name):
            asm_cont = find_global_asm_func(root_dir, c_file, asm_file_or_func_name)
            require_matching_func_name = True
        else:
            print("Could not open assembly file:", e, file=sys.stderr)
            sys.exit(1)

    ret = prune_asm(asm_cont)
    if require_matching_func_name and ret[0] != asm_file_or_func_name:
        # Safe-guard, since we currently only support one function per .s file.
        # Once that restriction is lifted it would be fine to return
        # asm_file_or_func_name as the function name here.
        print("GLOBAL_ASM file contains multiple functions.", file=sys.stderr)
        sys.exit(1)
    return ret


def create_directory(func_name: str) -> str:
    os.makedirs("nonmatchings/", exist_ok=True)
    ctr = 0
    while True:
        ctr += 1
        dirname = f"{func_name}-{ctr}" if ctr > 1 else func_name
        dirname = f"nonmatchings/{dirname}"
        try:
            os.mkdir(dirname)
            return dirname
        except FileExistsError:
            pass


def find_root_dir(filename: str, pattern: List[str]) -> Optional[str]:
    old_dirname = None
    dirname = os.path.abspath(os.path.dirname(filename))

    while dirname and (not old_dirname or len(dirname) < len(old_dirname)):
        for fname in pattern:
            if os.path.isfile(os.path.join(dirname, fname)):
                return dirname
        old_dirname = dirname
        dirname = os.path.dirname(dirname)

    return None


def fixup_build_command(
    parts: List[str], ignore_part: str
) -> Tuple[List[str], Optional[List[str]]]:
    res: List[str] = []
    skip_count = 0
    assembler = None
    last_o = len(parts) - 1 - parts[::-1].index("-o")
    for i, part in enumerate(parts):
        if skip_count > 0:
            skip_count -= 1
            continue
        if part in ["-MF", "-MT", "-MQ"] or i == last_o:
            skip_count = 1
            continue
        if part in ["-M", "-MM", "-MG", "-MP", "-MD", "-MMD", "-Mno-modules"]:
            continue
        if part == ignore_part:
            continue
        res.append(part)

    try:
        ind0 = min(
            i
            for i, arg in enumerate(res)
            if any(
                cmd in arg
                for cmd in [
                    "asm_processor",
                    "asm-processor",
                    "build.py",
                    "preprocess.py",
                ]
            )
        )
        ind1 = res.index("--", ind0 + 1)
        ind2 = res.index("--", ind1 + 1)
        compiler = res[ind0 + 1 : ind1]
        assembler = res[ind1 + 1 : ind2]
        compiler_args = res[ind2 + 1 :]
        while compiler and compiler[0].startswith("-"):
            flag = compiler.pop(0)
            if flag in (
                "--input-enc",
                "--output-enc",
                "--asm-prelude",
                "--convert-statics",
                "--keep-preprocessed",
            ):
                compiler.pop(0)
        res = compiler + compiler_args
    except ValueError:
        pass

    return res, assembler


def find_build_command_line(
    root_dir: str, c_file: str, make_flags: List[str], build_system: str
) -> Tuple[List[str], List[str]]:
    if build_system == "make":
        build_invocation = [
            make_cmd,
            "--always-make",
            "--dry-run",
            "--debug=j",
            "PERMUTER=1",
        ] + make_flags
    elif build_system == "ninja":
        build_invocation = ["ninja", "-t", "commands"] + make_flags
    else:
        print("Unknown build system '" + build_system + "'.")
        sys.exit(1)

    rel_c_file = os.path.relpath(c_file, root_dir)
    debug_output = (
        subprocess.check_output(build_invocation, cwd=root_dir)
        .decode("utf-8")
        .split("\n")
    )

    output: List[List[str]] = []
    close_match: str = ""

    assembler = DEFAULT_AS_CMDLINE
    for line in debug_output:
        while "//" in line:
            line = line.replace("//", "/")
        while "/./" in line:
            line = line.replace("/./", "/")
        if rel_c_file not in line:
            continue

        close_match = line
        parts = shlex.split(line)

        # extract actual command from 'bash -c "..."'
        if parts[0] == "bash" and "-c" in parts:
            for part in parts:
                if rel_c_file in part:
                    parts = shlex.split(part)
                    break

        if rel_c_file not in parts:
            continue
        if "-o" not in parts:
            continue
        if "-fsyntax-only" in parts:
            continue
        cmdline, asmproc_assembler = fixup_build_command(parts, rel_c_file)
        if asmproc_assembler:
            assembler = asmproc_assembler
        output.append(cmdline)

    if not output:
        close_extra = (
            "\nFound one possible candidate, but didn't match due to "
            "either spaces in paths, having -fsyntax-only, or missing an -o flag:\n"
            + close_match
            if close_match
            else ""
        )
        print(
            "Failed to find compile command from build script output. "
            f"Please ensure running '{' '.join(build_invocation)}' "
            f"contains a line with the string '{rel_c_file}'.{close_extra}",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(output) > 1:
        output_lines = "\n".join(map(formatcmd, output))
        print(
            f"Error: found multiple compile commands for {rel_c_file}:\n{output_lines}\n"
            f"Please modify the build script such that '{' '.join(build_invocation)}' "
            "produces a single compile command.",
            file=sys.stderr,
        )
        sys.exit(1)

    return output[0], assembler


PreserveMacros = Tuple[Pattern[str], Callable[[str], str]]


def build_preserve_macros(
    cwd: str, preserve_regex: Optional[str], settings: Mapping[str, object]
) -> Optional[PreserveMacros]:
    subdata = json_dict(settings, "preserve_macros")
    regexes: List[Tuple[re.Pattern[str], str]] = []
    for regex, value in subdata.items():
        assert isinstance(value, str)
        regexes.append((re.compile(f"^(?:{regex})$"), value))

    if preserve_regex == "" or (preserve_regex is None and not regexes):
        return None

    if preserve_regex is None:
        global_regex_text = "(?:" + ")|(?:".join(subdata.keys()) + ")"
    else:
        global_regex_text = preserve_regex
    global_regex = re.compile(f"^(?:{global_regex_text})$")

    def type_fn(macro: str) -> str:
        for regex, value in regexes:
            if regex.match(macro):
                return value
        return "int"

    return global_regex, type_fn


def preprocess_c_with_macros(
    cpp_command: List[str], cwd: str, preserve_macros: PreserveMacros
) -> Tuple[str, List[str]]:
    """Import C file, preserving function macros. Subroutine of import_c_file.

    Returns the source code and a list of preserved macros."""

    preserve_regex, preserve_type_fn = preserve_macros

    # Start by running 'cpp' in a mode that just processes ifdefs and includes.
    source = subprocess.check_output(
        cpp_command + ["-dD", "-fdirectives-only"], cwd=cwd, encoding="utf-8"
    )

    # Modify function macros that match preserved names so the preprocessor
    # doesn't touch them, and at the same time normalize their syntax. Some
    # of these instances may be in comments, but that's fine.
    def repl(match: Match[str]) -> str:
        name = match.group(1)
        after = "(" if match.group(2) == "(" else " "
        if preserve_regex.match(name):
            return f"_permuter define {name}{after}"
        else:
            return f"#define {name}{after}"

    source = re.sub(
        r"^\s*#\s*define\s+([a-zA-Z0-9_]+)([ \t\(]|$)",
        repl,
        source,
        flags=re.MULTILINE,
    )

    # Get rid of auto-inserted macros which the second cpp invocation will
    # warn about.
    source = re.sub(r"^#define __STDC_.*\n", "", source, flags=re.MULTILINE)

    # Now, run the preprocessor again for real.
    source = subprocess.check_output(
        CPP + STUB_FN_MACROS + ["-dD"], cwd=cwd, encoding="utf-8", input=source
    )

    # Finally, find all function-like defines that we hid (some might have
    # been comments, so we couldn't do this before), and construct fake
    # function declarations for them in a specially demarcated section of
    # the file. When the compiler runs, this section will be replaced by
    # the real defines and the preprocessor invoked once more.
    late_defines: List[Tuple[str, str]] = []
    helper_defines: Dict[str, str] = {}
    changing_helpers: Set[str] = set()
    lines: List[str] = []
    graph: Dict[str, Set[str]] = defaultdict(set)
    # Quoted tokens and formal parameters do not name macro dependencies.
    reg_token = re.compile(
        r"(?:u8|[uUL])?(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
        r"|(?P<identifier>[a-zA-Z_][a-zA-Z0-9_]*)"
    )
    for line in source.splitlines():
        is_helper = line.startswith("#define ")
        if is_helper:
            line = "_permuter define " + line[len("#define ") :]
        elif line.startswith("#undef "):
            changing_helpers.add(line.split()[1])
            continue
        is_macro = line.startswith("_permuter define ")
        params: Set[str] = set()
        ignore = False
        if "_permuter_ignore_line " in line:
            line = line.replace("_permuter_ignore_line ", "")
            ignore = True
        if is_macro:
            ind1 = line.find("(")
            ind2 = line.find(" ", len("_permuter define "))
            ind = min(ind1, ind2)
            if ind == -1:
                ind = len(line) if ind1 == ind2 == -1 else max(ind1, ind2)
            before = line[:ind]
            after = line[ind:]
            name = before.split()[2]
            if is_helper:
                # cpp reintroduces these built-ins after the first-pass filter.
                # Retaining them can cause redefinition warnings at compilation.
                if name.startswith("__STDC_"):
                    continue
                if name in helper_defines and helper_defines[name] != after:
                    changing_helpers.add(name)
                helper_defines[name] = after
            else:
                late_defines.append((name, after))
            if after.startswith("("):
                end_params = after.find(")")
                params.update(
                    re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", after[1:end_params])
                )
                if "..." in after[1:end_params]:
                    params.add("__VA_ARGS__")
                line = after[end_params + 1 :]
            else:
                line = after
        else:
            if ignore:
                encoded = b64encode(line.encode("utf-8")).decode("ascii")
                lines.append(f"#pragma _permuter b64literal {encoded}")
            else:
                lines.append(line)
            name = ""
        dependencies = graph[name]
        for m in reg_token.finditer(line):
            name2 = m.group("identifier")
            if name2 is None or name2 in params:
                continue
            has_wildcard = False
            if is_macro:
                wcbefore = line[: m.start()].rstrip().endswith("##")
                wcafter = line[m.end() :].lstrip().startswith("##")
                if wcbefore or wcafter:
                    dependencies.add(name2 + "*")
                    has_wildcard = True
            if not has_wildcard:
                dependencies.add(name2)

    used_by_nonmacro = graph[""]
    preserved_names = {name for name, _after in late_defines}

    def macro_closure(include_helpers: bool) -> Set[str]:
        used: Set[str] = set()
        queue = list(used_by_nonmacro & preserved_names)
        while queue:
            name = queue.pop()
            if name in used:
                continue
            used.add(name)
            if name.endswith("*"):
                wildcard = name[:-1]
                queue.extend(name2 for name2 in graph if wildcard in name2)
            elif include_helpers or name not in helper_defines:
                queue.extend(graph[name])
        return used

    # A preserved macro may refer to a helper defined later. Keep the helper
    # closure too, but only when moving it to the late-defines block cannot
    # expand identifiers left in ordinary source or change a dynamic macro.
    # Otherwise retain the existing preservation behavior for the whole closure.
    used_anywhere = macro_closure(include_helpers=True)
    helpers = used_anywhere & helper_defines.keys()
    if helpers & (changing_helpers | used_by_nonmacro):
        used_anywhere = macro_closure(include_helpers=False)
    else:
        late_defines.extend(
            (name, after) for name, after in helper_defines.items() if name in helpers
        )

    def get_decl(name: str, after: str) -> str:
        typ = preserve_type_fn(name)
        if after.startswith("("):
            return f"{typ} {name}();"
        else:
            return f"extern {typ} {name};"

    used_macros = [
        name
        for name, _after in late_defines
        if name in used_by_nonmacro and name in preserved_names
    ]

    return (
        "\n".join(
            ["#pragma _permuter latedefine start"]
            + [
                f"#pragma _permuter define {name}{after}"
                for (name, after) in late_defines
                if name in used_anywhere
            ]
            + [
                get_decl(name, after)
                for (name, after) in late_defines
                if name in used_by_nonmacro and name in preserved_names
            ]
            + ["#pragma _permuter latedefine end"]
            + lines
            + [""]
        ),
        used_macros,
    )


def import_c_file(
    compiler: List[str],
    cwd: str,
    in_file: str,
    preserve_macros: Optional[PreserveMacros],
) -> str:
    """Preprocess a C file into permuter-usable source.

    Prints preserved macros as a side effect.

    Returns source for base.c and compilable (macro-expanded) source."""
    in_file = os.path.relpath(in_file, cwd)
    include_next = 0
    cpp_command = CPP + [
        in_file,
        "-D__sgi",
        "-D_LANGUAGE_C",
        "-DNON_MATCHING",
        "-DNONMATCHING",
        "-DPERMUTER",
        "-D_MIPS_SZINT=32",
        "-D_MIPS_SZLONG=32",
    ]

    for arg in compiler:
        if include_next > 0:
            include_next -= 1
            cpp_command.append(arg)
            continue
        if arg in ["-D", "-U", "-I"]:
            cpp_command.append(arg)
            include_next = 1
            continue
        if (
            arg.startswith("-D")
            or arg.startswith("-U")
            or arg.startswith("-I")
            or arg in ["-nostdinc"]
        ):
            cpp_command.append(arg)

    try:
        if preserve_macros is None:
            # Simple codepath, should work even if the more complex one breaks.
            source = subprocess.check_output(
                cpp_command + STUB_FN_MACROS, cwd=cwd, encoding="utf-8"
            )
            macros: List[str] = []
        else:
            source, macros = preprocess_c_with_macros(cpp_command, cwd, preserve_macros)

    except subprocess.CalledProcessError as e:
        print(
            "Failed to preprocess input file, when running command:\n"
            + formatcmd(e.cmd),
            file=sys.stderr,
        )
        sys.exit(1)

    if macros:
        macro_str = "macros: " + ", ".join(macros)
    else:
        macro_str = "no macros"
    print(f"Preserving {macro_str}. Use --preserve-macros='<regex>' to override.")

    return source


def prune_source(
    source: str, should_prune: bool, func_name: str
) -> Tuple[str, Optional[str]]:
    """Normalize the source by round-tripping it through pycparser, and
    optionally reduce it to a smaller version that includes only the imported
    function and functions/struct/variables that it uses.

    Returns (source, compilable_source)."""
    try:
        ast = ast_util.parse_c(source, from_import=True)
        orig_fn, _ = ast_util.extract_fn(ast, func_name)
        if should_prune:
            try:
                ast_util.prune_ast(orig_fn, ast)
                source = ast_util.to_c_raw(ast)
            except Exception:
                print(
                    "Source minimization failed! "
                    "You could try --no-prune as a workaround."
                )
                raise
        return source, ast_util.to_c(ast, from_import=True)
    except CandidateConstructionFailure as e:
        print(e.message)
        if should_prune and "PERM_" in source:
            print(
                "Please put in PERM macros after import, otherwise source "
                "minimization does not work."
            )
        else:
            print("Proceeding anyway, but expect errors when permuting!")
        return source, None


def prune_and_separate_context(
    source: str, should_prune: bool, func_name: str
) -> Tuple[str, str]:
    """Normalize the source by round-tripping it through pycparser, optionally
    reduce it to a smaller version that includes only the imported function and
    functions/struct/variables that it uses, and split the result into source
    for the function itself, and the rest of the file (the "context").

    Returns (source, context)."""
    try:
        ast = ast_util.parse_c(source, from_import=True)
    except CandidateConstructionFailure as e:
        print(e.message)
        print("Unable to split context from source.")
        print("Using the entire source as context.")
        return "", ast_util.process_pragmas(source)

    try:
        orig_fn, ind = ast_util.extract_fn(ast, func_name)
    except CandidateConstructionFailure as e:
        print(e.message)
        print("Falling back to just removing function bodies.")
        context = ast_util.to_c(ast, from_import=True)
        return "", context

    if should_prune:
        try:
            ind = ast_util.prune_ast(orig_fn, ast)
        except Exception:
            print(
                "Source minimization failed! "
                "You could try --no-prune as a workaround."
            )
            raise
    del ast.ext[ind]
    source = ast_util.to_c(orig_fn, from_import=True)
    context = ast_util.to_c(ast, from_import=True)
    return source, context


def get_decompme_compiler_name(
    compiler: List[str], settings: Mapping[str, object], api_base: str
) -> str:
    decompme_settings = json_dict(settings, "decompme")
    compiler_mappings = json_dict(decompme_settings, "compilers")

    compiler_path = compiler[0]

    for path, compiler_name in compiler_mappings.items():
        assert isinstance(compiler_name, str)

        if fnmatch.fnmatch(compiler_path, path):
            return compiler_name

    available_ids: List[str] = []
    try:
        req = urllib.request.Request(
            f"{api_base}/api/compiler",
            headers={"User-Agent": "decomp-permuter"},
        )
        with urllib.request.urlopen(req) as f:
            json_data = json.load(f)
            available = json_dict(json_data, "compilers", allow_missing=False)
            available_ids = list(available.keys())
    except Exception as e:
        print(f"Failed to request available compilers from decomp.me:\n{e}")

    print()
    print(
        f'Unable to map compiler path "{compiler_path}" to something '
        "decomp.me understands."
    )
    trail = "permuter_settings.toml, where ... is one of: " + ", ".join(available_ids)
    if compiler_mappings:
        print(
            "Please add an entry: (wildcards allowed!)\n\n"
            f'"{compiler_path}" = "..."\n\n'
            f"to the [decompme.compilers] section of {trail}"
        )
    else:
        print(
            "Please add a section: (wildcards allowed!)\n\n"
            "[decompme.compilers]\n"
            f'"{compiler_path}" = "..."\n\n'
            f"to {trail}"
        )
    sys.exit(1)


def finalize_compile_command(cmdline: List[str]) -> str:
    quoted = [arg if arg == "|" else shlex.quote(arg) for arg in cmdline]
    ind = (quoted + ["|"]).index("|")
    return " ".join(quoted[:ind] + ['"$INPUT"'] + quoted[ind:] + ["-o", '"$OUTPUT"'])


def get_compiler_flags(settings: Mapping[str, object], cmdline: List[str]) -> str:
    decompme_settings = json_dict(settings, "decompme")
    flags = decompme_settings.get("flags")
    if flags:
        assert isinstance(flags, str)
        return flags
    flags = [b for a, b in zip(cmdline, cmdline[1:]) if a != "|" and b != "|"]
    return " ".join(shlex.quote(flag) for flag in flags)


def write_compile_command(compiler: List[str], cwd: str, out_file: str) -> None:
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n")
        f.write('INPUT="$(realpath "$1")"\n')
        f.write('OUTPUT="$(realpath "$3")"\n')
        f.write(f"cd {shlex.quote(cwd)}\n")
        f.write(finalize_compile_command(compiler))
    os.chmod(out_file, 0o755)


def write_asm(asm_prelude_file: Optional[str], asm_cont: str, out_file: str) -> None:
    asm_prelude_file = asm_prelude_file or DEFAULT_ASM_PRELUDE_FILE
    with open(asm_prelude_file, "r") as p:
        asm_prelude = p.read()
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(asm_prelude)
        f.write(asm_cont)


def compile_asm(assembler: List[str], cwd: str, in_file: str, out_file: str) -> None:
    in_file = os.path.abspath(in_file)
    out_file = os.path.abspath(out_file)
    cmdline = assembler + [in_file, "-o", out_file]
    try:
        subprocess.check_call(cmdline, cwd=cwd)
    except subprocess.CalledProcessError:
        print(
            f"Failed to assemble .s file, command line:\n{formatcmd(cmdline)}",
            file=sys.stderr,
        )
        sys.exit(1)


def compile_base(compile_script: str, source: str, c_file: str, out_file: str) -> None:
    if "PERM_" in source:
        print(
            "Cannot test-compile imported code because it contains PERM macros. "
            "It is recommended to put in PERM macros after import."
        )
        return
    escaped_c_file = json.dumps(c_file)
    source = "#line 1 " + escaped_c_file + "\n" + source
    compiler = Compiler(compile_script, show_errors=True, debug_mode=False)
    o_file = compiler.compile(source)
    if o_file:
        shutil.move(o_file, out_file)
    else:
        print("Warning: failed to compile .c file.")


def create_write_settings_toml(
    func_name: str,
    compiler_type: str,
    filename: str,
    objdump_command: Optional[str] = None,
) -> None:
    rand_weights = get_default_randomization_weights(compiler_type)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f'func_name = "{func_name}"\n')
        f.write(f'compiler_type = "{compiler_type}"\n')
        if objdump_command:
            f.write(f'objdump_command = "{objdump_command}"\n')
        f.write("\n")

        f.write("# uncomment lines below to customize randomization pass weights\n")
        f.write("# see --help=randomization-passes for descriptions\n")
        f.write("[weight_overrides]\n")
        for key, weight in rand_weights.items():
            f.write(f"# {key} = {weight}\n")


def write_to_file(cont: str, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(cont)


def main(arg_list: List[str]) -> None:
    parser = argparse.ArgumentParser(
        description="""Import a function for use with the permuter.
        Will create a new directory nonmatchings/<funcname>-<id>/."""
    )
    parser.add_argument(
        "c_file",
        help="""File containing the function.
        Assumes that the file can be built with 'make' to create an .o file.""",
    )
    parser.add_argument(
        "asm_file_or_func_name",
        metavar="{asm_file|func_name}",
        help="""File containing assembly for the function.
        Must start with 'glabel <function_name>' and contain no other functions.
        Alternatively, a function name can be given, which will be looked for in
        all GLOBAL_ASM blocks in the C file.""",
    )
    parser.add_argument(
        "make_flags",
        nargs="*",
        help="Arguments to pass to 'make'. PERMUTER=1 will always be passed.",
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep the directory on error."
    )
    settings_files = ", ".join(SETTINGS_FILES[:-1]) + " or " + SETTINGS_FILES[-1]
    parser.add_argument(
        "--preserve-macros",
        metavar="REGEX",
        dest="preserve_macros_regex",
        help=f"""Regex for which macros to preserve, or empty string for no macros.
        By default, this is read from {settings_files} in a parent directory of
        the imported file. Type information is also read from this file.""",
    )
    parser.add_argument(
        "--no-prune",
        dest="prune",
        action="store_false",
        help="""Don't minimize the source to keep only the imported function and
        functions/struct/variables that it uses. Normally this behavior is
        useful to make the permuter faster, but in cases where unrelated code
        affects the generated assembly asm it can be necessary to turn off.
        Note that regardless of this setting the permuter always removes all
        other functions by replacing them with declarations.""",
    )
    parser.add_argument(
        "--decompme",
        dest="decompme",
        action="store_true",
        help="""Upload the function to decomp.me to share with other people,
        instead of importing.""",
    )
    parser.add_argument(
        "--settings",
        dest="settings_file",
        metavar="SETTINGS_FILE",
        help="""Path to settings file.""",
    )
    args = parser.parse_args(arg_list)

    root_dir = find_root_dir(
        args.c_file, SETTINGS_FILES + ["Makefile", "makefile", "build.ninja"]
    )

    if not root_dir:
        print("Can't find root dir of project!", file=sys.stderr)
        sys.exit(1)

    settings: Mapping[str, object] = {}
    if args.settings_file:
        if os.path.exists(args.settings_file):
            with open(args.settings_file) as f:
                settings = toml.load(f)
        else:
            print("Can't find settings file!", file=sys.stderr)
            sys.exit(1)
    else:
        for filename in SETTINGS_FILES:
            filename = os.path.join(root_dir, filename)
            if os.path.exists(filename):
                with open(filename) as f:
                    settings = toml.load(f)
                break

    def get_setting(key: str) -> Optional[str]:
        value = settings.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            print(
                f"Value of {key} in settings.toml must be a string, but found: {value}"
            )
            sys.exit(1)
        return value

    build_system_raw = get_setting("build_system")
    build_system = build_system_raw or "make"
    compiler_str = get_setting("compiler_command") or ""
    assembler_str = get_setting("assembler_command") or ""
    asm_prelude_file = get_setting("asm_prelude_file")
    objdump_command = get_setting("objdump_command")
    if asm_prelude_file is not None:
        asm_prelude_file = os.path.join(root_dir, asm_prelude_file)
    make_flags = args.make_flags

    compiler_type = get_setting("compiler_type")
    if compiler_type is not None:
        print(f"Compiler type: {compiler_type}")
    else:
        compiler_type = "base"
        print(
            "Warning: Compiler type is missing from this project's permuter settings.\n"
            "Defaulting to base compiler randomization settings. For best permutation results,\n"
            "please set 'compiler_type' in this project's permuter_settings.toml."
        )

    func_name, asm_cont = parse_asm(root_dir, args.c_file, args.asm_file_or_func_name)
    print(f"Function name: {func_name}")

    if compiler_str or assembler_str:
        assert (
            build_system_raw is None
        ), "Must not specify both build system and compiler/assembler"
        compiler = shlex.split(compiler_str)
        assembler = shlex.split(assembler_str)
    else:
        compiler, assembler = find_build_command_line(
            root_dir, args.c_file, make_flags, build_system
        )

    print(f"Compiler: {finalize_compile_command(compiler)}")
    print(f'Assembler: {formatcmd(assembler)} "$INPUT" -o "$OUTPUT"')

    preserve_macros = build_preserve_macros(
        root_dir, args.preserve_macros_regex, settings
    )
    source = import_c_file(compiler, root_dir, args.c_file, preserve_macros)

    if args.decompme:
        api_base = os.environ.get("DECOMPME_API_BASE", "https://decomp.me")
        compiler_name = get_decompme_compiler_name(compiler, settings, api_base)
        source, context = prune_and_separate_context(source, args.prune, func_name)
        try:
            post_data = urllib.parse.urlencode(
                {
                    "name": func_name,
                    "target_asm": asm_cont,
                    "context": context,
                    "source_code": source,
                    "compiler": compiler_name,
                    "compiler_flags": get_compiler_flags(settings, compiler),
                    "diff_label": func_name,
                }
            ).encode("ascii")
            req = urllib.request.Request(
                f"{api_base}/api/scratch",
                data=post_data,
                headers={"User-Agent": "decomp-permuter"},
            )
            with urllib.request.urlopen(req) as f:
                resp = f.read()
                json_data: Dict[str, str] = json.loads(resp)
                if "slug" in json_data:
                    slug = json_data["slug"]
                    token = json_data.get("claim_token")
                    if token:
                        print(f"https://decomp.me/scratch/{slug}/claim?token={token}")
                    else:
                        print(f"https://decomp.me/scratch/{slug}")
                else:
                    error = json_data.get("error", resp)
                    print(f"Server error: {error}")
        except Exception as e:
            print(e)
        return

    source, compilable_source = prune_source(source, args.prune, func_name)

    dirname = create_directory(func_name)
    base_c_file = f"{dirname}/base.c"
    base_o_file = f"{dirname}/base.o"
    target_s_file = f"{dirname}/target.s"
    target_o_file = f"{dirname}/target.o"
    compile_script = f"{dirname}/compile.sh"
    settings_file = f"{dirname}/settings.toml"

    try:
        write_to_file(source, base_c_file)
        create_write_settings_toml(
            func_name, compiler_type, settings_file, objdump_command
        )
        write_compile_command(compiler, root_dir, compile_script)
        write_asm(asm_prelude_file, asm_cont, target_s_file)
        compile_asm(assembler, root_dir, target_s_file, target_o_file)
        if compilable_source is not None:
            compile_base(compile_script, compilable_source, base_c_file, base_o_file)
    except:
        if not args.keep:
            print(f"\nDeleting directory {dirname} (run with --keep to preserve it).")
            shutil.rmtree(dirname)
        raise

    print(f"\nDone. Imported into {dirname}")


if __name__ == "__main__":
    main(sys.argv[1:])
