from typing import Callable, Dict, Iterator, List, Match, Pattern, Tuple
import re

from .perm import (
    CombinePerm,
    ForceSamelinePerm,
    GeneralPerm,
    IgnorePerm,
    IntPerm,
    LineSwapPerm,
    LineSwapAstPerm,
    OncePerm,
    Perm,
    PretendPerm,
    RandomizerPerm,
    RootPerm,
    TextPerm,
    VarPerm,
)

# Consume entire C literals/comments before looking for macro syntax. Retain
# their original text; only punctuation and complete PERM identifiers are syntax.
_C_TEXT = r"""
    "(?:\\[\s\S]|[^"\\])*" | '(?:\\[\s\S]|[^'\\])*'
    | /\*[\s\S]*?\*/ | //(?:\\\r?\n|[^\n])*
    | (?P<unterminated>["']|/\*)
"""
_ARG_SYNTAX = re.compile(_C_TEXT + r"|(?P<syntax>[(),])", re.VERBOSE)
_MACRO_SYNTAX = re.compile(_C_TEXT + r"|(?P<macro>\bPERM_[A-Za-z0-9_]+)\(", re.VERBOSE)


def _syntax_matches(pattern: Pattern[str], text: str) -> Iterator[Match[str]]:
    for match in pattern.finditer(text):
        if match.lastgroup == "unterminated":
            raise Exception("Unterminated C literal or comment in PERM input")
        if match.lastgroup is not None:
            yield match


def _split_by_comma(text: str) -> List[str]:
    level = 0
    start = 0
    args: List[str] = []
    for match in _syntax_matches(_ARG_SYNTAX, text):
        c = match.group("syntax")
        if c == "," and level == 0:
            args.append(text[start : match.start()])
            start = match.end()
        else:
            if c == "(":
                level += 1
            elif c == ")":
                level -= 1
                assert level >= 0, "Bad nesting"
    assert level == 0, "Mismatched parentheses"
    args.append(text[start:])
    return args


def _split_args(text: str) -> List[Perm]:
    perm_args = [_rec_perm_parse(arg) for arg in _split_by_comma(text)]
    return perm_args


def _split_args_newline(text: str) -> List[Perm]:
    return [_rec_perm_parse(line) for line in text.split("\n") if line.strip()]


def _split_args_text(text: str) -> List[str]:
    perm_list = _split_args(text)
    res: List[str] = []
    for perm in perm_list:
        assert isinstance(perm, TextPerm)
        res.append(perm.text)
    return res


def _make_once_perm(text: str) -> OncePerm:
    args = _split_by_comma(text)
    if len(args) not in [1, 2]:
        raise Exception("PERM_ONCE takes 1 or 2 arguments")
    key = args[0].strip()
    value = _rec_perm_parse(args[-1])
    return OncePerm(key, value)


def _make_var_perm(text: str) -> VarPerm:
    args = _split_by_comma(text)
    if len(args) not in [1, 2]:
        raise Exception("PERM_VAR takes 1 or 2 arguments")
    var_name = _rec_perm_parse(args[0])
    value = _rec_perm_parse(args[1]) if len(args) == 2 else None
    return VarPerm(var_name, value)


PERM_FACTORIES: Dict[str, Callable[[str], Perm]] = {
    "PERM_GENERAL": lambda text: GeneralPerm(_split_args(text)),
    "PERM_ONCE": lambda text: _make_once_perm(text),
    "PERM_RANDOMIZE": lambda text: RandomizerPerm(_rec_perm_parse(text)),
    "PERM_FORCE_SAMELINE": lambda text: ForceSamelinePerm(_rec_perm_parse(text)),
    "PERM_VAR": lambda text: _make_var_perm(text),
    "PERM_LINESWAP_TEXT": lambda text: LineSwapPerm(_split_args_newline(text)),
    "PERM_LINESWAP": lambda text: LineSwapAstPerm(_split_args_newline(text)),
    "PERM_INT": lambda text: IntPerm(*map(int, _split_args_text(text))),
    "PERM_IGNORE": lambda text: IgnorePerm(_rec_perm_parse(text)),
    "PERM_PRETEND": lambda text: PretendPerm(_rec_perm_parse(text)),
}


def _consume_arg_parens(text: str) -> Tuple[str, str]:
    level = 0
    for match in _syntax_matches(_ARG_SYNTAX, text):
        c = match.group("syntax")
        if c == "(":
            level += 1
        elif c == ")":
            level -= 1
            if level == -1:
                return text[: match.start()], text[match.end() :]
    raise Exception("Failed to find closing parenthesis when parsing PERM macro")


def _rec_perm_parse(text: str) -> Perm:
    remain = text

    perms: List[Perm] = []
    while len(remain) > 0:
        match = next(_syntax_matches(_MACRO_SYNTAX, remain), None)

        # No match found; return remaining
        if match is None:
            text_perm = TextPerm(remain)
            perms.append(text_perm)
            break

        # Get perm type and args
        perm_type = match.group("macro")
        if perm_type not in PERM_FACTORIES:
            raise Exception("Unrecognized PERM macro: " + perm_type)
        between = remain[: match.start()]
        args, remain = _consume_arg_parens(remain[match.end() :])

        # Create text perm
        perms.append(TextPerm(between))

        # Create new perm
        perms.append(PERM_FACTORIES[perm_type](args))

    if len(perms) == 1:
        return perms[0]
    return CombinePerm(perms)


def perm_parse(text: str) -> Perm:
    ret = _rec_perm_parse(text)
    if RootPerm(ret).perm_count <= 1 and not ret.is_random():
        ret = RandomizerPerm(ret)
        print("No perm macros found. Defaulting to randomization.")
    ret = RootPerm(ret)
    if not ret.is_random():
        print(f"Will run for {ret.perm_count} iterations.")
    else:
        print(f"Will try {ret.perm_count} different base sources.")
    return ret
