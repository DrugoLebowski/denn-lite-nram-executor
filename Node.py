# Vendor
import pygraphviz as pgv

class Node(object):

    # Enum
    Register = 0
    Gate = 1

    def __init__(self, type: int, name: str, arity: int = -1):
        self.__type = type
        self.__name = name
        self.__arity = arity
        self.__nodes = []

    @property
    def type(self) -> int:
        return self.__type

    @type.setter
    def type(self, type: int) -> None:
        self.__type = type

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str) -> None:
        self.__name = name

    @property
    def nodes(self) -> list:
        return self.__nodes

    @nodes.setter
    def nodes(self, nodes: list) -> None:
        self.__nodes = nodes

    def add_node(self, node):
        self.__nodes.append(node)

    def check_validity(self) -> bool:
        for node in self.__nodes:
            if not node.check_validity():
                self.__nodes.remove(node)
        if self.name != "Write" and "R'" not in self.name and len(self.__nodes) == 0:
            return False
        return True

    def check_instance(self, name: str) -> bool:
        if self.__name == name:
            return True
        else:
            if len(self.__nodes) == 0:
                return False
            else:
                for node in self.__nodes:
                    if node.check_instance(name):
                        return True
                return False
