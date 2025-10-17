import pytest
import numpy as np

from primel.tree import ExpressionTree, Node, simplify_tree


@pytest.fixture
def sample_data() -> np.ndarray:
    """Provides a sample numpy array for evaluation."""
    return np.array([1, 2, 3, 4, 5])


class TestExpressionTree:
    """Tests for the ExpressionTree class methods."""

    def test_tree_creation_and_str(self):
        """Test basic tree creation and its string representation."""
        tree = ExpressionTree("add", lambda a, b: a + b, 2)
        tree.add_node("x0", lambda x: x, 0)
        tree.add_node("const", 2.0, 0)

        assert str(tree).startswith("add")
        assert "x0" in str(tree)
        assert "const" in str(tree)
        assert len(tree.nodes) == 3

    def test_subtree_size(self):
        """Test the _subtree_size method with nested children."""
        # Represents mul(add(x0, 1), x1)
        nodes_list = [
            Node("mul", np.multiply, 2),
            Node("add", np.add, 2),
            Node("x0", lambda x: x[:, 0], 0),
            Node("const", lambda x: 1, 0),
            Node("x1", lambda x: x[:, 1], 0),
        ]
        tree = ExpressionTree.init_from_list(nodes_list)

        assert tree._subtree_size(0) == 5  # whole tree mul(...)
        assert tree._subtree_size(1) == 3  # subtree add(...)
        assert tree._subtree_size(2) == 1  # leaf x0
        assert tree._subtree_size(4) == 1  # leaf x1

    def test_evaluate(self, sample_data):
        """Test the evaluate method on a simple expression."""
        # Represents (x + 5)
        tree = ExpressionTree("add", np.add, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("const", 5.0, 0)

        result = tree.evaluate(sample_data)
        expected = np.array([6, 7, 8, 9, 10])
        np.testing.assert_array_equal(result, expected)

    def test_evaluate_nested(self, sample_data):
        """Test the evaluate method on a nested expression."""
        # Represents add(mul(x, 2), sub(x, 3))
        tree = ExpressionTree("add", np.add, 2)
        tree.add_node("mul", np.multiply, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("const2", 2.0, 0)
        tree.add_node("sub", np.subtract, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("const3", 3.0, 0)

        result = tree.evaluate(sample_data)
        expected = (sample_data * 2) + (sample_data - 3)
        np.testing.assert_array_equal(result, expected)

    def test_replace_subtree(self, sample_data):
        """Test replacing a subtree with a new node."""
        # Start with add(mul(x, 2), 5)
        tree = ExpressionTree("add", np.add, 2)
        tree.add_node("mul", np.multiply, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("const2", 2.0, 0)
        tree.add_node("const5", 5.0, 0)

        # Replace mul(x, 2) subtree (at index 1) with just x
        tree.replace_subtree(1, "x", lambda x: x, 0)

        # Expected tree is now add(x, 5)
        assert len(tree.nodes) == 3
        assert tree.nodes[0].name == "add"
        assert tree.nodes[1].name == "x"
        assert tree.nodes[2].name == "const5"

        result = tree.evaluate(sample_data)
        expected = sample_data + 5
        np.testing.assert_array_equal(result, expected)

    def test_replace_node_with_child(self):
        # Tree is add(mul(x, 2), 5)
        tree = ExpressionTree("add", np.add, 2)
        tree.add_node("mul", np.multiply, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("const2", 2.0, 0)
        tree.add_node("const5", 5.0, 0)

        tree.replace_node_with_child(0, 0)

        assert len(tree.nodes) == 3
        assert tree.nodes[0].name == "mul"


class TestSimplifyTree:
    """Tests for the heuristic simplification rules."""

    def test_simplify_log(self, sample_data):
        """Test rule: log(x) -> x"""
        tree = ExpressionTree("log", np.log, 1)
        tree.add_node("x", lambda x: x, 0)

        simplify_tree(tree, sample_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "x"

    def test_simplify_trig_small_range(self, sample_data):
        """Test rule: sin(x) -> x for small input range."""
        tree = ExpressionTree("sin", np.sin, 1)
        tree.add_node("x", lambda x: x, 0)
        X = np.linspace(0, 1, 5)  # Range is 1, which is < 2*pi

        simplify_tree(tree, X)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "x"

    def test_simplify_trig_large_range(self, sample_data):
        """Test rule: sin(x) should not change for large input range."""
        tree = ExpressionTree("sin", np.sin, 1)
        tree.add_node("x", lambda x: x, 0)
        X = np.linspace(0, 10, 5)  # Range is 10, which is > 2*pi

        simplify_tree(tree, X)

        assert len(tree.nodes) == 2
        assert tree.nodes[0].name == "sin"

    def test_simplify_sub_x_x(self, sample_data):
        """Test rule: sub(x, x) -> 0"""
        tree = ExpressionTree("sub", np.subtract, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("x", lambda x: x, 0)

        simplify_tree(tree, sample_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "constant"
        assert tree.nodes[0].value == 0

    def test_simplify_div_x_x(self, sample_data):
        """Test rule: div(x, x) -> 1"""
        tree = ExpressionTree("div", np.divide, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("x", lambda x: x, 0)

        simplify_tree(tree, sample_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "constant"
        assert tree.nodes[0].value == 1

    def test_simplify_op_with_constant_right(self, sample_data):
        """Test rule: op(x, const) -> x"""
        tree = ExpressionTree("add", np.add, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("const", 5, 0)

        simplify_tree(tree, sample_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "x"

    def test_simplify_op_with_constant_left(self, sample_data):
        """Test rule: op(const, y) -> y"""
        tree = ExpressionTree("mul", np.multiply, 2)
        tree.add_node("const", 5.0, 0)
        tree.add_node("y", lambda y: y, 0)

        simplify_tree(tree, sample_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "y"

    def test_simplify_nested(self, sample_data):
        """Test simplification in a nested tree structure."""
        # Represents add(mul(x, 2), sub(x, 2))
        nodes_list = [
            Node("add", np.add, 2),
            Node("mul", np.multiply, 2),
            Node("x", lambda x: x, 0),
            Node("const2", 2.0, 0),
            Node("sub", np.subtract, 2),
            Node("x", lambda x: x, 0),
            Node("const2", 2.0, 0),
        ]
        tree = ExpressionTree.init_from_list(nodes_list)

        simplify_tree(tree, sample_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "x"
