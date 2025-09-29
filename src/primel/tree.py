from typing import Any, Callable, Self


class Node:
    def __init__(self: Self, value: Any, repr_func: Callable | None = None):
        self.value = value
        self.children = []
        self.repr_func = repr_func

    def __str__(self, level=0):
        if self.repr_func is not None:
            ret = "\t" * level + self.repr_func(self.value) + "\n"
        else:
            ret = "\t" * level + repr(self.value) + "\n"

        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


def reduce(root: Node) -> Node:
    pass
