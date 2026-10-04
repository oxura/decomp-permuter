from random import Random
import unittest
import warnings

from perm_pycparser import c_ast as ca

from src import ast_util
from src.ast_types import build_typemap, decayed_expr_type, expr_type, TypeMap
from src.candidate import Candidate
from src.perm.perm import EvalState
from src.randomizer import RANDOMIZATION_PASSES, perm_temp_for_expr, Region


class TestImplicitFunctions(unittest.TestCase):
    def typemap(self, source: str) -> TypeMap:
        ast = ast_util.parse_c(source)
        fn = ast_util.extract_fn(ast, "target")[0]
        return build_typemap(ast, fn)

    def test_undeclared_call_returns_int(self) -> None:
        typemap = self.typemap("int target(void) { return missing(1); }")
        call = ca.FuncCall(ca.ID("missing"), ca.ExprList([ca.Constant("int", "1")]))
        self.assertEqual(ast_util.to_c_raw(expr_type(call, typemap)), "int")
        self.assertIsInstance(expr_type(call.name, typemap), ca.FuncDecl)
        self.assertIsInstance(decayed_expr_type(call.name, typemap), ca.PtrDecl)
        self.assertNotIn("missing", typemap.local_vars)

    def test_nested_calls_in_initializer(self) -> None:
        typemap = self.typemap(
            "int target(void) { int value = outer(inner()); return value; }"
        )
        for name in ["inner", "outer"]:
            with self.subTest(name=name):
                self.assertEqual(
                    ast_util.to_c_raw(
                        expr_type(ca.FuncCall(ca.ID(name), None), typemap)
                    ),
                    "int",
                )

    def test_explicit_types_are_preserved(self) -> None:
        sources = [
            "double declared(void); double target(void) { return declared(); }",
            "double target(void) { double declared(void); return declared(); }",
            "double target(double (*declared)(void)) { return declared(); }",
            "typedef double (*Callback)(void); "
            "double target(Callback declared) { return declared(); }",
            "double target(void) { return declared(); } double declared(void);",
        ]
        for source in sources:
            with self.subTest(source=source):
                typemap = self.typemap(source)
                self.assertEqual(
                    ast_util.to_c_raw(
                        expr_type(ca.FuncCall(ca.ID("declared"), None), typemap)
                    ),
                    "double",
                )

    def test_struct_function_pointer_is_preserved(self) -> None:
        typemap = self.typemap(
            "struct Callbacks { double (*callback)(void); }; "
            "double target(struct Callbacks *callbacks) { return callbacks->callback(); }"
        )
        call = ca.FuncCall(
            ca.StructRef(ca.ID("callbacks"), "->", ca.ID("callback")), None
        )
        self.assertEqual(ast_util.to_c_raw(expr_type(call, typemap)), "double")
        self.assertNotIn("callback", typemap.var_types)

    def test_explicit_non_function_is_not_replaced(self) -> None:
        typemap = self.typemap("int declared; int target(void) { return declared(); }")
        with self.assertRaisesRegex(AssertionError, "call to non-function"):
            expr_type(ca.FuncCall(ca.ID("declared"), None), typemap)

    def test_undeclared_variable_is_not_a_function(self) -> None:
        typemap = self.typemap("int target(void) { return missing; }")
        with self.assertRaises(KeyError):
            expr_type(ca.ID("missing"), typemap)

    def test_other_function_calls_are_not_inferred(self) -> None:
        typemap = self.typemap(
            "int other(void) { return unrelated(); } int target(void) { return missing(); }"
        )
        self.assertNotIn("unrelated", typemap.var_types)
        self.assertIsInstance(typemap.var_types["missing"], ca.FuncDecl)

    def test_typemap_does_not_change_source(self) -> None:
        ast = ast_util.parse_c("int target(void) { return missing(); }")
        fn = ast_util.extract_fn(ast, "target")[0]
        before = ast_util.to_c_raw(ast)
        build_typemap(ast, fn)
        self.assertEqual(ast_util.to_c_raw(ast), before)

    def test_randomizer_can_extract_undeclared_call(self) -> None:
        for seed in range(20):
            with self.subTest(seed=seed):
                ast = ast_util.parse_c("int target(int x) { return missing(x) + 1; }")
                fn = ast_util.extract_fn(ast, "target")[0]
                ast_util.normalize_ast(fn, ast)
                perm_temp_for_expr(
                    fn,
                    ast,
                    ast_util.compute_node_indices(fn),
                    Region.unbounded(),
                    Random(seed),
                )


class TestImplicitFunctionWarnings(unittest.TestCase):
    def setUp(self) -> None:
        Candidate._cached_shared_ast.cache_clear()

    def tearDown(self) -> None:
        Candidate._cached_shared_ast.cache_clear()

    def candidate(self, source: str) -> Candidate:
        weights = {method.__name__: 1.0 for method in RANDOMIZATION_PASSES}
        return Candidate.from_source(source, EvalState(), "target", weights, rng_seed=0)

    def test_warning_before_randomization_without_adding_declaration(self) -> None:
        source = "int target(void) { return missing() + missing(); }"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            candidate = self.candidate(source)
        self.assertEqual(len(caught), 1)
        self.assertIn("missing", str(caught[0].message))
        self.assertIn("int", str(caught[0].message))
        self.assertEqual(
            candidate.get_source(), ast_util.to_c(ast_util.parse_c(source))
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.candidate(source)
            candidate.randomize_ast()
        self.assertEqual(caught, [])

    def test_each_missing_function_warns_once(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            self.candidate("int target(void) { return zeta(alpha()) + zeta(); }")
        self.assertEqual(len(caught), 2)
        self.assertIn("alpha", str(caught[0].message))
        self.assertIn("zeta", str(caught[1].message))

    def test_source_variants_and_cache_eviction_do_not_repeat_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(2):
                for n in range(17):
                    self.candidate(f"int target(void) {{ return missing() + {n}; }}")
        self.assertEqual(len(caught), 1)

    def test_explicit_warning_filter_is_respected(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("ignore", RuntimeWarning)
            self.candidate("int target(void) { return missing(); }")
        self.assertEqual(caught, [])

    def test_known_functions_and_pointers_do_not_warn(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.candidate(
                "int known(void); int target(int (*callback)(void)) "
                "{ return known() + callback(); }"
            )
        self.assertEqual(caught, [])
