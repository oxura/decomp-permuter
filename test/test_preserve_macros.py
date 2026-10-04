import importlib
import itertools
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from src import ast_util

importer = importlib.import_module("import")


class TestPreserveMacros(unittest.TestCase):
    def preprocess_source(self, source, *, header=""):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.c"
            path.write_text(source)
            (Path(directory) / "helper.h").write_text(header)
            command = importer.CPP + [str(path)]
            expected = subprocess.check_output(command, cwd=directory, encoding="utf-8")
            actual, macros = importer.preprocess_c_with_macros(
                command, directory, (re.compile("^A$"), lambda _: "int")
            )
        self.assertEqual(macros, ["A"])
        return expected, actual

    def assert_equivalent(self, expected, actual):
        compilable = ast_util.process_pragmas(actual)
        expanded = subprocess.check_output(
            importer.CPP, input=compilable, encoding="utf-8"
        )
        # Ignore formatting outside literals, including formatting added by
        # the parser/generator in the pruning test.
        token = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\w+|\S'
        self.assertEqual(re.findall(token, expanded), re.findall(token, expected))
        subprocess.run(
            ["gcc", "-Werror", "-x", "c", "-fsyntax-only", "-"],
            input=compilable,
            encoding="utf-8",
            check=True,
            capture_output=True,
        )

    def test_object_helpers_in_any_definition_order(self):
        definitions = ["#define A B", "#define B C", "#define C 42"]
        for ordered in itertools.permutations(definitions):
            with self.subTest(definitions=ordered):
                expected, actual = self.preprocess_source(
                    "\n".join(ordered) + "\nint f(void) { return A; }\n"
                )
                self.assert_equivalent(expected, actual)

    def test_nested_function_helpers(self):
        expected, actual = self.preprocess_source(
            "#define A(x) B(x)\n"
            "#define B(y) C(y)\n"
            "#define C(z) ((z) + VALUE)\n"
            "#define VALUE 1\n"
            "int f(int x) { return A(x); }\n"
        )
        self.assert_equivalent(expected, actual)
        self.assertNotIn("int B();", actual)
        self.assertNotIn("extern int VALUE;", actual)

    def test_stringification_helper(self):
        expected, actual = self.preprocess_source(
            "#define A(x) B(x)\n"
            "#define B(x) #x\n"
            "const char *f(void) { return A(some_identifier); }\n"
        )
        self.assert_equivalent(expected, actual)

    def test_token_pasting_helpers(self):
        for first in ("#define A(x) B(x)\n#define B(x) X ## x", "#define A(x) X ## x"):
            with self.subTest(first=first):
                expected, actual = self.preprocess_source(
                    first + "\n#define X1 42\nint f(void) { return A(1); }\n"
                )
                self.assert_equivalent(expected, actual)
                self.assertIn("#pragma _permuter define X1 42", actual)
                self.assertNotIn("__STDC_", actual)

    def test_parameters_are_not_helper_dependencies(self):
        expected, actual = self.preprocess_source(
            "#define A(x) B(x)\n"
            "#define B(x) ((x) + 1)\n"
            "int f(int x) { return A(x); }\n"
            "#define x 42\n"
        )
        self.assert_equivalent(expected, actual)
        self.assertNotIn("#pragma _permuter define x ", actual)

    def test_variadic_helpers(self):
        expected, actual = self.preprocess_source(
            "#define A(...) B(__VA_ARGS__)\n"
            "#define B(...) (0 + __VA_ARGS__)\n"
            "int f(void) { return A(42); }\n"
        )
        self.assert_equivalent(expected, actual)

    def test_quoted_tokens_and_comments_are_not_dependencies(self):
        for value, return_type in (
            ("'B'", "int"),
            ('"B X"', "const char *"),
            ("42 /* B */", "int"),
        ):
            with self.subTest(value=value):
                expected, actual = self.preprocess_source(
                    "#define A " + value + "\n"
                    "int B;\n" + return_type + " f(void) { return A; }\n"
                    "#define B 42\n"
                    "#define X 1\n"
                )
                self.assert_equivalent(expected, actual)
                self.assertNotIn("#pragma _permuter define B ", actual)
                self.assertNotIn("#pragma _permuter define X ", actual)

    def test_unrelated_helpers_are_pruned(self):
        expected, actual = self.preprocess_source(
            "#define A B\n"
            "#define B 42\n"
            "#define UNUSED X\n"
            "#define X UNUSED\n"
            "int f(void) { return A; }\n"
        )
        self.assert_equivalent(expected, actual)
        self.assertNotIn("#pragma _permuter define UNUSED ", actual)
        self.assertNotIn("#pragma _permuter define X ", actual)

    def test_conditional_and_included_helpers(self):
        expected, actual = self.preprocess_source(
            '#define A B\n#include "helper.h"\nint f(void) { return A; }\n',
            header="#if 0\n#define B 0\n#else\n#define B 42\n#endif\n",
        )
        self.assert_equivalent(expected, actual)

    def test_helper_directive_inside_expression(self):
        expected, actual = self.preprocess_source(
            "#define A B\nint f(void) { return\n#define B 42\nA; }\n"
        )
        self.assert_equivalent(expected, actual)

    def test_identical_redefinitions(self):
        expected, actual = self.preprocess_source(
            "#define A B\n#define B 42\n#define B 42\nint f(void) { return A; }\n"
        )
        self.assert_equivalent(expected, actual)
        self.assertEqual(actual.count("#pragma _permuter define B 42"), 1)

    def test_ordinary_helper_expansion_is_unchanged(self):
        expected, actual = self.preprocess_source(
            "#define A B\n#define B 42\nint f(void) { return A + B; }\n"
        )
        self.assert_equivalent(expected, actual)

    def test_helper_type_dependencies_survive_pruning(self):
        expected, actual = self.preprocess_source(
            "typedef int Value;\n"
            "#define A B\n"
            "#define B ((Value)42)\n"
            "int f(void) { return A; }\n"
        )
        pruned, compilable = importer.prune_source(actual, True, "f")
        self.assertIsNotNone(compilable)
        self.assertIn("typedef int Value;", pruned)
        self.assert_equivalent(expected, pruned)

    def test_unsafe_helpers_keep_existing_preservation_behavior(self):
        # These dynamic/colliding cases were not supported by preservation.
        # Do not make them worse by activating helpers in already expanded C.
        sources = [
            "#define A B\n#define B 42\nint f(void) { return A; }\n#undef B\nint B;\n",
            "#define A B\nint B;\n#define B 42\nint f(void) { return A; }\n",
            "#define A B\n#define B 1\nint f(void) { return A; }\n#undef B\n#define B 2\n",
            "int B;\n#define A B\n#define B (B + 1)\nint f(void) { return A + B; }\n",
        ]
        for source in sources:
            with self.subTest(source=source):
                _expected, actual = self.preprocess_source(source)
                self.assertIn("#pragma _permuter define A B", actual)
                self.assertNotIn("#pragma _permuter define B ", actual)
                self.assertNotIn("#undef ", actual)

    def test_unsafe_helper_disables_the_whole_new_closure(self):
        _expected, actual = self.preprocess_source(
            "int C;\n#define A (B + C)\n#define B 1\n#define C 2\n"
            "int f(void) { return A; }\n"
        )
        self.assertNotIn("#pragma _permuter define B ", actual)
        self.assertNotIn("#pragma _permuter define C ", actual)

    def test_previously_resolved_helpers_stay_resolved_after_undef(self):
        expected, actual = self.preprocess_source(
            "#define B 42\n#define A B\nint f(void) { return A; }\n#undef B\nint B;\n"
        )
        self.assert_equivalent(expected, actual)
        self.assertNotIn("#pragma _permuter define B ", actual)


if __name__ == "__main__":
    unittest.main()
