import order as od

from .utils import is_collection


# Extend od.Variable to include label and branches
class Var(od.Variable):
    def __init__(self, name, branch, label, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.label = label
        self.branch = branch if is_collection(branch) else [branch]
        if kwargs.get("expression", None) is None:
            self.expression = lambda x: x


def get_var_input_branches(variables):
    branches = set()
    for var in variables:
        if is_collection(var.branch):
            branches.update(var.branch)
        else:
            branches.add(var.branch)
    return list(branches)
