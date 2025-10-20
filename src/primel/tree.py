from typing import Any, Callable, Self

import numpy as np


class Node:
    def __init__(
        self: Self,
        name: str,
        value: Callable | float | int,
        arity: int,
        repr_func: Callable | None = None,
    ):
        self.name = name
        self.value = value
        self.arity = arity
        self.repr_func = repr_func

    def __str__(self, level=0):
        ret = "\t" * level + self.name + "\n"

        return ret


class ExpressionTree:
    def __init__(
        self: Self,
        root_name: str,
        root_value: Any,
        root_arity: int,
        repr_func: Callable | None = None,
    ):
        self.nodes = [Node(root_name, root_value, root_arity, repr_func)]

    @classmethod
    def init_from_list(cls: type[Self], nodes: list[Node]) -> Self:
        tree = cls.__new__(cls)
        tree.nodes = nodes
        return tree

    def __str__(self: Self):
        def _str(node_index: int, level: int) -> str:
            node = self.nodes[node_index]
            ret = node.__str__(level)
            child_index = node_index + 1
            for _ in range(node.arity):
                ret += _str(child_index, level + 1)
                child_index += self._subtree_size(child_index)
            return ret

        return _str(0, 0)

    def _subtree_size(self: Self, node_index: int) -> int:
        node = self.nodes[node_index]
        size = 1
        child_index = node_index + 1
        for _ in range(node.arity):
            child_size = self._subtree_size(child_index)
            size += child_size
            child_index += child_size
        return size

    def add_node(
        self: Self,
        name: str,
        value: Callable | float | int,
        arity: int,
        repr_func: Callable | None = None,
    ):
        self.nodes.append(Node(name, value, arity, repr_func))

    def replace_node_with_child(
        self: Self,
        node_index: int,
        replaced_by_child: int,
    ):
        # Calculate the size of the entire subtree being replaced.
        size_to_replace = self._subtree_size(node_index)

        # Find the starting index of the child subtree that will replace the parent.
        child_start_index = node_index + 1
        for _ in range(replaced_by_child):
            child_start_index += self._subtree_size(child_start_index)

        # Get the full subtree of the child.
        child_subtree_size = self._subtree_size(child_start_index)
        child_subtree = self.nodes[
            child_start_index : child_start_index + child_subtree_size
        ]

        # Replace the original subtree slice with the child's full subtree.
        self.nodes[node_index : node_index + size_to_replace] = child_subtree

    def remove_subtree(self: Self, node_index: int):
        size = self._subtree_size(node_index)
        del self.nodes[node_index : node_index + size]

    def replace_subtree(
        self: Self,
        node_index: int,
        name: str,
        value: Any,
        arity: int,
        repr_func: Callable | None = None,
    ):
        size = self._subtree_size(node_index)
        self.nodes[node_index : node_index + size] = [
            Node(name, value, arity, repr_func)
        ]

    def evaluate(self: Self, X: np.ndarray, index: int = 0) -> np.ndarray:
        """
        Evaluates the expression tree at a given node index.

        This implementation is optimized to avoid repeated calculations of subtree
        sizes, which was a major performance bottleneck in the previous version.
        It uses a single recursive function that evaluates a node and
        simultaneously computes the size of the subtree rooted at that node.
        This avoids the need for separate, repeated calls to a `_subtree_size`
        method during evaluation.
        """

        def _eval(node_index: int) -> tuple[np.ndarray, int]:
            """Recursively evaluates a subtree and returns the result and its size."""
            node = self.nodes[node_index]
            size = 1
            if node.arity == 0:
                if callable(node.value):
                    val = node.value(X)
                else:
                    val = np.full(X.shape[0], node.value)
                return val, size
            else:
                child_values = []
                child_index = node_index + 1
                for _ in range(node.arity):
                    child_val, child_size = _eval(child_index)
                    child_values.append(child_val)
                    child_index += child_size
                    size += child_size
                return node.value(*child_values), size

        val, _ = _eval(index)
        return val


def simplify_tree(tree: ExpressionTree, X: np.ndarray) -> None:
    """
    Heuristic simplification of the expression tree. This is done to make the
    function behave more nicely when early stopping is used.

    The simplification rules are as follows:
    Arity 1 Replace op(x) with x given these rules:
    - replace sin(x), cos(x), tan(x) with x if for x [a, b] b-a < 2pi
    - replace x^2 with x if x >= 0
    - always replace log, sqrt
    Arity 2 Replace op(x, y) with x or y given these rules:
    - replace op(x, constant) with x
    - replace op(constant, y) with y
    - replace sub(x, x) with 0
    - replace div(x, x) with 1
    - replace add(x, x) with x
    - replace mul(x, x) with x if x >= 0
    """

    # TODO: implement simplification rules
    index = len(tree.nodes) - 1
    for node in tree.nodes[::-1]:
        if node.arity == 1:
            if node.name in {"log", "sqrt"}:
                tree.replace_node_with_child(index, 0)
            elif node.name in {"sin", "cos", "tan"}:
                result = tree.evaluate(X, index + 1)
                if result.max() - result.min() < 2 * np.pi:
                    tree.replace_node_with_child(index, 0)
            elif node.name == "square":
                result = tree.evaluate(X, index + 1)
                if (result >= 0).all():
                    tree.replace_node_with_child(index, 0)

        elif node.arity == 2:
            if node.name not in {"add", "sub", "mul", "div"}:
                continue

            left = tree.nodes[index + 1]
            right_index = index + 1 + tree._subtree_size(index + 1)
            right = tree.nodes[right_index]

            if left.arity == 0 and isinstance(left.value, (int, float)):
                tree.replace_node_with_child(index, 1)
            elif right.arity == 0 and isinstance(right.value, (int, float)):
                tree.replace_node_with_child(index, 0)
            elif left.name == right.name and left.arity == right.arity:
                if node.name == "sub":
                    tree.replace_subtree(index, "constant", 0, 0)
                elif node.name == "div":
                    tree.replace_subtree(index, "constant", 1, 0)
                elif node.name == "add":
                    tree.replace_node_with_child(index, 0)
                elif node.name == "mul":
                    result = tree.evaluate(X, index + 1)
                    if (result >= 0).all():
                        tree.replace_node_with_child(index, 0)

        index -= 1
