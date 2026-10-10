import contextlib
import io
import os
from typing import List
import unittest

from src.compiler import Compiler
from src.perm.parse import perm_parse
from src.perm.perm import EvalState


class TestLiteralMacroParsing(unittest.TestCase):
    def variants(self, source: str) -> List[str]:
        with contextlib.redirect_stdout(io.StringIO()):
            root = perm_parse(source)
        return [root.evaluate(i, EvalState()) for i in range(root.perm_count)]

    def assert_compiles(self, variants: List[str]) -> None:
        compiler = Compiler("test/compile.sh", show_errors=True, debug_mode=False)
        for value in variants:
            obj = compiler.compile(value, show_errors=True)
            self.assertIsNotNone(obj)
            if obj:
                os.unlink(obj)

    def test_string_comma(self) -> None:
        variants = self.variants(
            'const char *foo(void) { return PERM_GENERAL("a,b", "c"); }'
        )
        self.assertEqual(len(variants), 2)
        self.assert_compiles(variants)
        self.assertIn('"a,b"', variants[0])

    def test_character_comma(self) -> None:
        variants = self.variants("int foo(void) { return PERM_GENERAL(',', 0); }")
        self.assertEqual(len(variants), 2)
        self.assert_compiles(variants)
        self.assertIn("','", variants[0])

    def test_closing_parenthesis_character(self) -> None:
        variants = self.variants("int foo(void) { return PERM_GENERAL(')', 0); }")
        self.assertEqual(len(variants), 2)
        self.assert_compiles(variants)

    def test_opening_parenthesis_character(self) -> None:
        variants = self.variants("int foo(void) { return PERM_GENERAL('(', 0); }")
        self.assertEqual(len(variants), 2)
        self.assert_compiles(variants)

    def test_macro_spelling_inside_literal(self) -> None:
        source = 'const char *foo(void) { return "PERM_GENERAL(a, b)"; }'
        variants = self.variants(source)
        self.assertEqual(len(variants), 1)
        self.assertIn(source, variants[0])
        self.assert_compiles(variants)

    def test_unknown_macro_spelling_inside_literal(self) -> None:
        source = 'const char *foo(void) { return "PERM_UNKNOWN(a)"; }'
        variants = self.variants(source)
        self.assertEqual(len(variants), 1)
        self.assertIn(source, variants[0])
        self.assert_compiles(variants)

    def test_escaped_quote_inside_literal(self) -> None:
        source = r'const char *foo(void) { return PERM_GENERAL("a\",b", "c"); }'
        variants = self.variants(source)
        self.assertEqual(len(variants), 2)
        self.assert_compiles(variants)

    def test_escaped_backslashes_and_character_quotes(self) -> None:
        for source in (
            r'const char *foo(void) { return PERM_GENERAL("a\\", ")"); }',
            r"int foo(void) { return PERM_GENERAL('\\', '\''); }",
            "int foo(void) { return PERM_GENERAL('\"', 0); }",
        ):
            with self.subTest(source=source):
                variants = self.variants(source)
                self.assertEqual(len(variants), 2)
                self.assert_compiles(variants)

    def test_comments_are_not_macro_syntax(self) -> None:
        for comment in (
            '/* PERM_UNKNOWN("(,)") and an unmatched quote: " */',
            '// PERM_UNKNOWN("(,)") and an unmatched quote: "\n',
        ):
            with self.subTest(comment=comment):
                source = "int foo(void) { " + comment + " return PERM_GENERAL(1, 2); }"
                variants = self.variants(source)
                self.assertEqual(len(variants), 2)
                self.assert_compiles(variants)
                self.assertTrue(all("PERM_UNKNOWN" in value for value in variants))

    def test_macro_names_must_be_complete_identifiers(self) -> None:
        source = (
            "int PREFIX_PERM_GENERAL(int a, int b); "
            'const char *PERM_text = "("; '
            "int foo(void) { return PREFIX_PERM_GENERAL(1, 2); }"
        )
        variants = self.variants(source)
        self.assertEqual(len(variants), 1)
        self.assertIn(source, variants[0])
        self.assert_compiles(variants)

    def test_continued_line_comments(self) -> None:
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=newline):
                source = (
                    "// continued comment \\"
                    + newline
                    + "PERM_UNKNOWN(1)"
                    + newline
                    + "int foo(void) { return PERM_GENERAL(1, 2); }"
                )
                variants = self.variants(source)
                self.assertEqual(len(variants), 2)
                self.assertTrue(all("PERM_UNKNOWN(1)" in value for value in variants))
                self.assert_compiles(variants)

    def test_unterminated_literals_and_comments_raise(self) -> None:
        for source in (
            'const char *foo(void) { return "PERM_GENERAL(a, b); }',
            'const char *foo(void) { return PERM_GENERAL("a,b, 0); }',
            "int foo(void) { return PERM_GENERAL(', 0); }",
            "/* PERM_GENERAL(1, 2)",
        ):
            with self.subTest(source=source):
                with self.assertRaisesRegex(Exception, "Unterminated C"):
                    self.variants(source)

    def test_nested_string_choices(self) -> None:
        variants = self.variants(
            'const char *foo(void) { return PERM_GENERAL("a,b", '
            'PERM_GENERAL("PERM_UNKNOWN(x)", ")")); }'
        )
        self.assertEqual(len(variants), 3)
        self.assert_compiles(variants)
        self.assertIn('"a,b"', variants[0])
        self.assertIn('"PERM_UNKNOWN(x)"', variants[1])
        self.assertIn('")"', variants[2])

    def test_documented_comma_escape_is_retained(self) -> None:
        variants = self.variants(
            'const char *foo(void) { return PERM_GENERAL("a(,)b", "c"); }'
        )
        self.assertEqual(len(variants), 2)
        self.assertIn('"a,b"', variants[0])
        self.assert_compiles(variants)

    def test_nested_macro_control(self) -> None:
        variants = self.variants(
            "int foo(void) { return PERM_GENERAL(1, PERM_GENERAL(2, 3)); }"
        )
        self.assertEqual(len(variants), 3)
        self.assert_compiles(variants)

    def test_unknown_actual_macro_control(self) -> None:
        with self.assertRaisesRegex(Exception, "Unrecognized PERM macro"):
            self.variants("int foo(void) { return PERM_UNKNOWN(1); }")


if __name__ == "__main__":
    unittest.main()
