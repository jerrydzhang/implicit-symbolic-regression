from typing import Any, List

from primel.tree import Node


def repr_func(value: Any) -> str:
    if hasattr(value, "name"):
        return value.name
    return repr(value)


def build_tree(program: List) -> Node:
    program = program[:]

    def _build(program):
        node_value = program.pop(0)
        node = Node(node_value, repr_func=repr_func)

        if hasattr(node_value, "arity"):
            for _ in range(node_value.arity):
                node.children.append(_build(program))

        return node

    return _build(program)


def unbuild_tree(node: Node) -> List:
    result = [node.value]
    for child in node.children:
        result.extend(unbuild_tree(child))
    return result
