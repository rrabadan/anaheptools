from collections.abc import Callable
from typing import Any

import order as od


class Var(od.Variable):
    """Extended Variable class with name, description, label and input branches information.

    Args:
        name: Variable name
        description: Optional description of the variable
        label: Optional Human-readable label for plotting
        input_branches: Optional Single branch name or list of input branch names
        expression: Optional Function to transform the input data
        **kwargs: Additional arguments passed to od.Variable
    """

    def __init__(
        self,
        name: str,
        description: str | None = None,
        label: str | None = None,
        input_branches: str | list[str] | None = None,
        expression: Callable[[Any], Any] | None = None,
        *args,
        **kwargs,
    ):
        # Remove branch and expression from kwargs to avoid conflicts
        kwargs.pop("input_branches", None)
        kwargs.pop("description", None)
        kwargs.pop("label", None)
        kwargs.pop("expression", None)

        super().__init__(name, *args, **kwargs)

        self.description = description or name
        self.label = label or name

        # Handle input branches assignment
        if input_branches is None:
            self.input_branches = [name]  # Default to variable name
        elif isinstance(input_branches, str):
            self.input_branches = [input_branches]  # Single branch as list
        else:
            self.input_branches = list(input_branches)

        # Handle expression assignment
        self.expression = expression or (lambda x: x)  # Default to identity function

    def get_input_branches(self) -> list[str]:
        """Get list of input branch names needed for this variable."""
        return self.input_branches.copy()  # Return copy to prevent external modification

    def get_branches(self) -> list[str]:
        """Alias for get_input_branches() for consistency."""
        return self.get_input_branches()

    def set_input_branches(self, *branches: str) -> None:
        """Set the input branches for this variable."""
        if not branches:
            raise ValueError("At least one branch name must be provided.")
        self.input_branches = list(set(branches))

    def validate_branches(self, available_branches: list[str]) -> bool:
        """Check if all required branches are available."""
        return all(branch in available_branches for branch in self.input_branches)

    def is_computed(self) -> bool:
        """Check if this variable requires computation (has non-identity expression)."""
        # This is tricky to check perfectly, but we can do a simple check
        try:
            return self.expression.__name__ != "<lambda>" or str(self.expression) != "lambda x: x"
        except Exception:
            return True  # Assume computed if we can't determine

    def __repr__(self) -> str:
        """String representation of the variable."""
        # branches_str = ", ".join(self.branch)
        return f"Var(name='{self.name}', description='{self.description}')"

    @classmethod
    def simple(cls, name: str, description: str | None = None, label: str | None = None) -> "Var":
        """Create a simple variable with minimal configuration."""
        return cls(
            name=name,
            description=description or name,
            label=label or name.replace("_", " ").title(),
        )


def get_var_input_branches(variables: list[Var]) -> list[str]:
    """Get all unique branch names needed for a list of variables.

    Args:
        variables: List of Var instances

    Returns:
        List of unique branch names
    """
    branches = set()
    for var in variables:
        branches.update(var.get_input_branches())

    # Filter out None values if they somehow get through
    return [branch for branch in branches if branch is not None]
