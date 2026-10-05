import os
from random import Random
import unittest
from unittest.mock import patch

from perm_pycparser import c_ast as ca

from src import ast_util
from src.compiler import Compiler
from src import randomizer


class TestLocalTypeDeclarations(unittest.TestCase):
    SOURCES = {
        "struct": """
            int foo(int a) {
                struct Tag { int member; };
                struct Tag value;
                value.member = a;
                return value.member + 1;
            }
        """,
        "union": """
            int foo(int a) {
                union Tag { int member; unsigned other; };
                union Tag value;
                value.member = a;
                return value.member + 1;
            }
        """,
        "enum": "int foo(int a) { enum Tag { MEMBER = 2 }; return a + MEMBER; }",
        "anonymous_enum": "int foo(int a) { enum { MEMBER = 2 }; return a + MEMBER; }",
        "anonymous_struct": "int foo(int a) { struct { int member; }; return a + 1; }",
        "anonymous_union": "int foo(int a) { union { int member; }; return a + 1; }",
        "forward_struct": "int foo(int a) { struct Tag; return a + 1; }",
        "forward_union": "int foo(int a) { union Tag; return a + 1; }",
        "initialized_variable": """
            int foo(int a) {
                struct Tag;
                int value = a + 1;
                return value + 1;
            }
        """,
        "shared_tag_member_variable_name": """
            int foo(int a) {
                struct value { int value; };
                struct value item;
                int value = a;
                item.value = value;
                return item.value + 1;
            }
        """,
        "anonymous_type_with_variable": """
            int foo(int a) {
                struct { int member; } value;
                value.member = a;
                return value.member + 1;
            }
        """,
        "nested_block": """
            int foo(int a) {
                int value;
                {
                    enum Tag { MEMBER = 2 };
                    value = a + MEMBER;
                }
                return value;
            }
        """,
    }

    def permute(self, source: str) -> str:
        ast = ast_util.parse_c(source)
        fn, _ = ast_util.extract_fn(ast, "foo")
        ast_util.normalize_ast(fn, ast)
        declarations = []

        class Visitor(ca.NodeVisitor):
            def visit_Decl(self, node: ca.Decl) -> None:
                if node.name is None:
                    declarations.append((node, ast_util.to_c(node)))
                self.generic_visit(node)

        Visitor().visit(fn)
        randomizer.perm_temp_for_expr(
            fn,
            ast,
            ast_util.compute_node_indices(fn),
            randomizer.Region.unbounded(),
            Random(0),
        )
        result = ast_util.to_c(ast)
        self.assertIn("new_var", result)
        for declaration, original in declarations:
            self.assertEqual(ast_util.to_c(declaration), original)
        return result

    def test_standalone_declarations(self) -> None:
        for name, source in self.SOURCES.items():
            with self.subTest(name=name):
                result = self.permute(source)
                ast_util.parse_c(result)

    def test_transformed_sources_compile(self) -> None:
        compiler = Compiler("test/compile.sh", show_errors=True, debug_mode=False)
        for name, source in self.SOURCES.items():
            with self.subTest(name=name):
                obj = compiler.compile(self.permute(source), show_errors=True)
                self.assertIsNotNone(obj)
                if obj:
                    os.unlink(obj)

    def test_only_variable_names_are_reuse_candidates(self) -> None:
        source = """
            int foo(int a) {
                struct Tag { int member; };
                enum OtherTag { MEMBER = 2 };
                typedef int Alias;
                int value = a;
                return value + MEMBER;
            }
        """
        with patch(
            "src.randomizer.maybe_reuse_var", wraps=randomizer.maybe_reuse_var
        ) as reuse:
            self.permute(source)
        reuse.assert_called_once()
        self.assertEqual(reuse.call_args[0][0], "value")

    def test_nested_declaration_stays_in_its_scope(self) -> None:
        result = self.permute(self.SOURCES["nested_block"])
        ast = ast_util.parse_c(result)
        fn, _ = ast_util.extract_fn(ast, "foo")
        assert fn.body.block_items is not None
        nested = [stmt for stmt in fn.body.block_items if isinstance(stmt, ca.Compound)]
        self.assertEqual(len(nested), 1)
        assert nested[0].block_items is not None
        enum = nested[0].block_items[0]
        assert isinstance(enum, ca.Decl)
        self.assertIsNone(enum.name)
        self.assertIsInstance(enum.type, ca.Enum)

    def test_randomizer_accepts_local_type_declaration(self) -> None:
        ast = ast_util.parse_c(self.SOURCES["enum"])
        fn, _ = ast_util.extract_fn(ast, "foo")
        ast_util.normalize_ast(fn, ast)
        weights = {
            method.__name__: float(method is randomizer.perm_temp_for_expr)
            for method in randomizer.RANDOMIZATION_PASSES
        }
        randomizer.Randomizer(weights, rng_seed=0).randomize(ast, "foo")
        self.assertIn("new_var", ast_util.to_c(ast))

    def test_declaration_without_expressions_declines(self) -> None:
        for declaration in (
            "struct Tag { int member; };",
            "union Tag { int member; };",
            "enum Tag { MEMBER = 2 };",
            "struct Tag;",
        ):
            with self.subTest(declaration=declaration):
                ast = ast_util.parse_c("void foo(void) { " + declaration + " }")
                fn, _ = ast_util.extract_fn(ast, "foo")
                with self.assertRaises(randomizer.RandomizationFailure):
                    randomizer.perm_temp_for_expr(
                        fn,
                        ast,
                        ast_util.compute_node_indices(fn),
                        randomizer.Region.unbounded(),
                        Random(0),
                    )


if __name__ == "__main__":
    unittest.main()
