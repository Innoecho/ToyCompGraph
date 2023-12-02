from graphviz import Digraph
from collections import deque

from variable import Variable


def get_node_attr(node: Variable) -> dict:
    if node.is_leaf:
        node_attr = {"color": "#FF9933", "style": "filled", "fontcolor": "white", "shape": "rectangle"}
    else:
        node_attr = {"color": "#3399FF", "style": "filled", "fontcolor": "white", "shape": "rectangle"}

    return node_attr


def view_graph(node: Variable, file_name: str = "./graph", label_type: int = 0) -> None:
    dot = Digraph(format="png")

    queue = deque()
    visited = set()

    queue.append(node)
    visited.add(node)
    dot.node(name=node.get_name(), label=node.get_label(label_type), **get_node_attr(node))

    while len(queue) > 0:
        cur_node = queue.popleft()
        for prev_node in cur_node.prev:
            if prev_node not in visited:
                visited.add(prev_node)
                queue.append(prev_node)
                dot.node(name=prev_node.get_name(), label=prev_node.get_label(label_type), **get_node_attr(prev_node))

            dot.edge(prev_node.get_name(), cur_node.get_name(), color="#000000")

    dot.render(filename=file_name, cleanup=True)
