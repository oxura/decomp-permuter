import unittest
from random import Random
from unittest.mock import patch

from perm_pycparser import c_ast as ca

from src import ast_util
from src.randomizer import (
    RandomizationFailure,
    Region,
    compute_read_locations,
    compute_write_locations,
    find_var_reads,
    perm_expand_expr,
)


class LoopAccessTests(unittest.TestCase):
    def parse(self, source):
        ast = ast_util.parse_c(source)
        fn = ast.ext[-1]
        return ast, fn, ast_util.compute_node_indices(fn)

    def expand(self, ast, fn, indices, read):
        target = indices.starts[read]

        def choose(candidates):
            self.assertIn(target, candidates)
            return target

        random = Random(0)
        with patch.object(random, "choice", side_effect=choose), patch(
            "src.randomizer.random_bool", side_effect=[True, False]
        ):
            perm_expand_expr(fn, ast, indices, Region.unbounded(), random)

    def assert_rejected(self, source, get_read):
        ast, fn, indices = self.parse(source)
        before = ast_util.to_c(ast)
        with self.assertRaises(RandomizationFailure):
            self.expand(ast, fn, indices, get_read(fn))
        self.assertEqual(ast_util.to_c(ast), before)

    def test_for_condition_does_not_use_initial_value(self):
        self.assert_rejected(
            "void f(int n) { int i; for (i = 0; i < n; i++) {} }",
            lambda fn: fn.body.block_items[1].cond.left,
        )

    def test_for_declaration_condition_does_not_use_initial_value(self):
        self.assert_rejected(
            "void f(int n) { for (int i = 0; i < n; i++) {} }",
            lambda fn: fn.body.block_items[0].cond.left,
        )

    def test_while_condition_does_not_use_initial_value(self):
        self.assert_rejected(
            "void f(int n) { int i = 0; while (i < n) { i++; } }",
            lambda fn: fn.body.block_items[1].cond.left,
        )

    def test_do_while_condition_does_not_use_initial_value(self):
        self.assert_rejected(
            "void f(int n) { int i = 0; do { i++; } while (i < n); }",
            lambda fn: fn.body.block_items[1].cond.left,
        )

    def test_read_after_loop_does_not_use_last_syntactic_write(self):
        self.assert_rejected(
            "void use(int); void f(int n) { int i = 0; "
            "while (n--) { i = 3; } use(i); }",
            lambda fn: fn.body.block_items[2].args.exprs[0],
        )

    def test_for_body_does_not_use_upcoming_increment_assignment(self):
        self.assert_rejected(
            "void use(int); void f(int n) { int x = 0; "
            "for (; n--; x = 2) { use(x); } }",
            lambda fn: fn.body.block_items[1].stmt.block_items[0].args.exprs[0],
        )

    def test_do_body_does_not_use_upcoming_condition_assignment(self):
        self.assert_rejected(
            "void use(int); void f(int n) { int x = 0; "
            "do { use(x); } while ((x = 2) < n); }",
            lambda fn: fn.body.block_items[1].stmt.block_items[0].args.exprs[0],
        )

    def test_effectful_value_is_not_moved_into_a_loop(self):
        self.assert_rejected(
            "int get(void); void use(int); void f(int n) { int x = get(); "
            "while (n--) { use(x); } }",
            lambda fn: fn.body.block_items[1].stmt.block_items[0].args.exprs[0],
        )

    def test_for_increment_can_use_this_iterations_body_assignment(self):
        ast, fn, indices = self.parse(
            "void use(int); void f(int n) { int x; " "for (; n--; use(x)) { x = 3; } }"
        )
        loop = fn.body.block_items[1]
        self.expand(ast, fn, indices, loop.next.args.exprs[0])
        self.assertEqual(loop.next.args.exprs[0].value, "3")
        self.assertIsInstance(loop.stmt.block_items[0], ca.Assignment)

    def test_do_condition_can_use_this_iterations_body_assignment(self):
        ast, fn, indices = self.parse(
            "void f(int n) { int x; do { x = 3; } while (x < n); }"
        )
        loop = fn.body.block_items[1]
        self.expand(ast, fn, indices, loop.cond.left)
        self.assertEqual(loop.cond.left.value, "3")
        self.assertIsInstance(loop.stmt.block_items[0], ca.Assignment)

    def test_nested_loop_write_reaches_outer_condition(self):
        self.assert_rejected(
            "void f(int n, int m) { int i = 0; "
            "while (i < n) { for (int j = 0; j < m; j++) { i++; } } }",
            lambda fn: fn.body.block_items[1].cond.left,
        )

    def test_write_needed_at_loop_entry_is_kept(self):
        ast, fn, indices = self.parse(
            "void use(int); void f(int n) { int i = 0; "
            "use(i); while (i < n) { i++; } }"
        )
        self.expand(ast, fn, indices, fn.body.block_items[1].args.exprs[0])
        self.assertEqual(fn.body.block_items[0].init.value, "0")
        self.assertEqual(fn.body.block_items[1].args.exprs[0].value, "0")

    def test_write_needed_on_next_iteration_is_kept(self):
        ast, fn, indices = self.parse(
            "void use(int); void f(int n) { int i = 0; "
            "while (i < n) { i = 3; use(i); } }"
        )
        body = fn.body.block_items[1].stmt
        self.expand(ast, fn, indices, body.block_items[1].args.exprs[0])
        self.assertIsInstance(body.block_items[0], ca.Assignment)
        self.assertEqual(body.block_items[1].args.exprs[0].value, "3")

    def test_for_initializer_is_not_a_loop_carried_write(self):
        ast, fn, indices = self.parse(
            "void use(int); void f(int n) { int i; " "for (i = 3; n; n--) { use(i); } }"
        )
        loop = fn.body.block_items[1]
        self.assertEqual(
            compute_write_locations(fn, indices)["i"],
            [indices.starts[fn.body.block_items[0]], indices.starts[loop.init]],
        )
        self.expand(ast, fn, indices, loop.stmt.block_items[0].args.exprs[0])
        self.assertEqual(loop.stmt.block_items[0].args.exprs[0].value, "3")
        self.assertIsInstance(loop.init, ca.Assignment)

    def test_loop_boundaries_are_ordered_read_before_write(self):
        _, fn, indices = self.parse(
            "void f(int n) { int i; for (i = 0; i < n; i++) {} }"
        )
        loop = fn.body.block_items[1]
        reads = compute_read_locations(fn, indices)["i"]
        writes = compute_write_locations(fn, indices)["i"]
        self.assertIn(indices.ends[loop.init], reads)
        self.assertIn(indices.ends[loop.init] + 1, writes)
        self.assertIn(indices.ends[loop] - 1, reads)
        self.assertIn(indices.ends[loop], writes)
        self.assertEqual(reads, sorted(set(reads)))
        self.assertEqual(writes, sorted(set(writes)))

    def test_assignment_rhs_and_compound_lhs_are_reads(self):
        _, fn, _ = self.parse("void f(int x, int y) { int out; out = x; out += y; }")
        self.assertEqual([node.name for node in find_var_reads(fn)], ["x", "out", "y"])

    def test_self_update_rhs_does_not_use_its_own_write(self):
        self.assert_rejected(
            "int f(void) { int x = 1; x = x + 1; return x; }",
            lambda fn: fn.body.block_items[1].rvalue.left,
        )

    def test_read_after_self_update_does_not_repeat_the_update(self):
        self.assert_rejected(
            "int f(void) { int x = 1; x = x + 1; return x; }",
            lambda fn: fn.body.block_items[2].expr,
        )

    def test_independent_assignment_rhs_can_be_expanded(self):
        ast, fn, indices = self.parse(
            "int f(void) { int x = 3; int y; y = x; return y; }"
        )
        self.expand(ast, fn, indices, fn.body.block_items[2].rvalue)
        self.assertEqual(fn.body.block_items[2].rvalue.value, "3")
        self.assertIsNone(fn.body.block_items[0].init)

    def test_assignment_rhs_keeps_a_loop_carried_write_alive(self):
        ast, fn, indices = self.parse(
            "void use(int); void f(int n) { int i = 0; int out; "
            "while (n--) { out = i; i = 3; use(i); } }"
        )
        body = fn.body.block_items[2].stmt
        self.expand(ast, fn, indices, body.block_items[2].args.exprs[0])
        self.assertIsInstance(body.block_items[1], ca.Assignment)
        self.assertEqual(body.block_items[2].args.exprs[0].value, "3")

    def test_unreplaceable_reads_are_not_selected(self):
        ast, fn, indices = self.parse(
            "void use(int); void f(int n) { int i = 0; "
            "while (n--) { i++; i += 2; use(i); } }"
        )
        loop = fn.body.block_items[1]
        body = loop.stmt.block_items
        forbidden = {
            indices.starts[loop],
            indices.ends[loop] - 1,
            indices.starts[loop.cond.expr],
            indices.starts[body[0].expr],
            indices.starts[body[1].lvalue],
        }

        def choose(candidates):
            self.assertTrue(candidates)
            self.assertTrue(forbidden.isdisjoint(candidates))
            return candidates[0]

        random = Random(0)
        with patch.object(random, "choice", side_effect=choose):
            try:
                perm_expand_expr(fn, ast, indices, Region.unbounded(), random)
            except RandomizationFailure:
                pass

    def test_straight_line_expansion_still_removes_unused_write(self):
        ast, fn, indices = self.parse("int f(void) { int x = 3; return x; }")
        self.expand(ast, fn, indices, fn.body.block_items[1].expr)
        self.assertIsNone(fn.body.block_items[0].init)
        self.assertEqual(fn.body.block_items[1].expr.value, "3")


if __name__ == "__main__":
    unittest.main()
