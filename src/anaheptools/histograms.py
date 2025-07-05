from collections.abc import Callable
from typing import Any

import hist
import matplotlib.pyplot as plt
import mplhep
import numpy as np
from hist import Hist

from .utils import is_collection


def hist1d_from_array(
    arr: dict[str, Any],
    branches: str | list[str],
    name: str,
    label: str,
    range_: tuple[float, float],
    bins: int,
    transform: Callable | None = None,
    weight: str | np.ndarray | None = None,
    **kwargs,
) -> Hist:
    """
    Create a histogram from array data.

    Args:
        arr: Dictionary-like object containing branch data (e.g., from uproot)
        branches: Single branch name or list of branch names to use
        name: Name identifier for the histogram
        label: Human-readable label for axis
        range_: Tuple of (min, max) values for histogram range
        bins: Number of bins for the histogram
        transform: Optional function to transform the input data
        weight: Optional weights (branch name or array)
        **kwargs: Additional histogram configuration
            - overflow: Include overflow bin (default: False)
            - underflow: Include underflow bin (default: False)
            - storage: Histogram storage type (default: hist.storage.Weight())

    Returns:
        Configured histogram filled with data

    Raises:
        ValueError: If branches are missing from array or range is invalid
        TypeError: If transform function signature is incompatible
    """
    # Input validation
    if not isinstance(branches, str | list):
        raise TypeError("branches must be string or list of strings")

    if isinstance(branches, str):
        branches = [branches]

    # Check if all branches exist in the array
    # missing_branches = [b for b in branches if b not in arr]
    # if missing_branches:
    #     raise ValueError(f"Missing branches in array: {missing_branches}")

    # Validate range
    if len(range_) != 2 or range_[0] >= range_[1]:
        raise ValueError(f"Invalid range: {range_}. Must be (min, max) with min < max")

    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")

    # Extract configuration
    overflow = kwargs.pop("overflow", False)
    underflow = kwargs.pop("underflow", False)
    storage = kwargs.pop("storage", hist.storage.Weight())

    # Create histogram
    h = Hist(
        hist.axis.Regular(bins, range_[0], range_[1], overflow=overflow, underflow=underflow),
        storage=storage,
    )

    # Set metadata
    h.label = label
    h.name = name

    # Prepare data
    try:
        if transform is not None:
            # Apply transformation
            branch_data = [arr[b] for b in branches]
            if len(branches) == 1:
                data = transform(branch_data[0])
            else:
                data = transform(*branch_data)
            data = (data,)
        else:
            # Use data directly
            if len(branches) == 1:
                data = (arr[branches[0]],)
            else:
                # For multiple branches without transform, take the first one
                # or you might want to raise an error here
                data = (arr[branches[0]],)
    except Exception as e:
        raise RuntimeError(f"Error applying transform or accessing data: {e}") from None

    # Handle weights
    weight_array = None
    if weight is not None:
        if isinstance(weight, str):
            if weight not in arr:
                raise ValueError(f"Weight branch '{weight}' not found in array")
            weight_array = arr[weight]
        else:
            weight_array = weight

    # Fill histogram
    try:
        h.fill(*data, weight=weight_array)
    except Exception as e:
        raise RuntimeError(f"Error filling histogram: {e}") from None

    return h


def hist1d_from_var(
    var, arr: dict[str, Any], weight: str | np.ndarray | None = None, **kwargs
) -> Hist:
    """
    Create a histogram from a Var object and input array.

    Args:
        var: Variable object with required attributes:
            - input_branches (or branch): branch name(s) in the input array
            - name: histogram identifier
            - label: axis label
            - x_min, x_max: histogram range
            - n_bins: number of bins
            - expression: optional transformation function
        arr: Dictionary-like object containing branch data
        weight: Optional weights (branch name or array)
        **kwargs: Additional histogram configuration passed to hist_from_array

    Returns:
        Configured histogram filled with data

    Raises:
        AttributeError: If required variable attributes are missing
        ValueError: If variable configuration is invalid
    """
    # Check for required attributes
    required_attrs = ["name", "label", "x_min", "x_max", "n_bins"]
    missing_attrs = [attr for attr in required_attrs if not hasattr(var, attr)]
    if missing_attrs:
        raise AttributeError(f"Variable missing required attributes: {missing_attrs}")

    # Get branches (support both old 'branch' and new 'input_branches')
    if hasattr(var, "input_branches"):
        branches = var.input_branches
    elif hasattr(var, "branch"):
        branches = var.branch
    else:
        raise AttributeError("Variable must have 'input_branches' or 'branch' attribute")

    # Ensure branches is a list
    if not is_collection(branches):
        branches = [branches]

    # Get transformation function
    transform = getattr(var, "expression", None)

    # Validate range
    if var.x_min >= var.x_max:
        raise ValueError(f"Invalid variable range: x_min={var.x_min}, x_max={var.x_max}")

    return hist1d_from_array(
        arr=arr,
        branches=branches,
        name=var.name,
        label=var.label,
        range_=(var.x_min, var.x_max),
        bins=var.n_bins,
        transform=transform,
        weight=weight,
        **kwargs,
    )


def quick_hist1d(
    arr: dict[str, Any],
    branch: str,
    bins: int = 50,
    range_: tuple[float, float] | None = None,
    weight: str | np.ndarray | None = None,
    **kwargs,
) -> Hist:
    """
    Quick histogram creation with automatic range detection.

    Args:
        arr: Dictionary-like object containing branch data
        branch: Branch name to histogram
        bins: Number of bins (default: 50)
        range_: Optional range tuple, auto-detected if None
        weight: Optional weights
        **kwargs: Additional configuration
    """
    if branch not in arr:
        raise ValueError(f"Branch '{branch}' not found in array")

    data = arr[branch]

    # Auto-detect range if not provided
    if range_ is None:
        if len(data) == 0:
            raise ValueError("Cannot auto-detect range for empty data")
        range_ = (float(np.min(data)), float(np.max(data)))
        # Add small padding
        padding = (range_[1] - range_[0]) * 0.05
        range_ = (range_[0] - padding, range_[1] + padding)

    return hist1d_from_array(
        arr=arr,
        branches=branch,
        name=branch,
        label=branch.replace("_", " ").title(),
        range_=range_,
        bins=bins,
        weight=weight,
        **kwargs,
    )


def plot_hist1d(
    ax: plt.Axes,
    h: Hist,
    logy: bool = False,
    show_stats: bool = False,
    show_overflow: bool = False,
    show_underflow: bool = False,
    **kwargs,
) -> None:
    """
    Plot a histogram on the given axes with extensive customization options.

    Args:
        ax: Matplotlib axes to plot on
        h: Histogram object to plot
        logy: Use logarithmic y-scale (default: False)
        show_stats: Display statistics box (entries, mean, std) (default: False)
        show_overflow: Include overflow in flow parameter (default: False)
        show_underflow: Include underflow in flow parameter (default: False)
        **kwargs: Additional plotting options:
            # Style options
            - histtype: "step", "fill", "stepfilled" (default: "step")
            - color: Plot color (default: "black")
            - alpha: Transparency (default: 1.0)
            - linewidth/lw: Line width (default: 1.5)
            - linestyle/ls: Line style (default: "-")

            # Label and legend options
            - label: Legend label (default: histogram name or None)
            - show_label_in_legend: Include label in legend (default: True if label provided)

            # Data options
            - density: Normalize to density (default: False)
            - flow: Flow handling "hint", "show", "sum", "none" (default: auto-determined)

            # Axis options
            - ylabel: Y-axis label (default: "Candidates" or "Density")
            - xlabel: X-axis label (default: histogram label)
            - ylim_bottom: Y-axis bottom limit (default: 0)
            - ylim_top: Y-axis top limit (default: auto)
            - xlim: X-axis limits tuple (default: auto)

            # Grid and styling
            - grid: Show grid (default: False)
            - grid_alpha: Grid transparency (default: 0.3)

    Raises:
        ValueError: If histogram is empty or invalid
        TypeError: If histogram type is not supported
    """
    # Input validation
    if h is None:
        raise ValueError("Histogram cannot be None")

    if not hasattr(h, "values"):
        raise TypeError("Object must be a histogram with 'values' method")

    # Check if histogram is empty
    if h.sum() == 0:
        print(f"Warning: Histogram '{getattr(h, 'name', 'unnamed')}' is empty")

    # Extract style parameters
    histtype = kwargs.pop("histtype", "step")
    color = kwargs.pop("color", "black")
    alpha = kwargs.pop("alpha", 1.0)
    linewidth = kwargs.pop("linewidth", kwargs.pop("lw", 1.5))
    linestyle = kwargs.pop("linestyle", kwargs.pop("ls", "-"))

    # Extract label parameters
    label = kwargs.pop("label", getattr(h, "name", None))
    show_label_in_legend = kwargs.pop("show_label_in_legend", label is not None)

    # Extract data parameters
    density = kwargs.pop("density", False)

    # Determine flow parameter
    flow = kwargs.pop("flow", None)
    if flow is None:
        if show_overflow and show_underflow:
            flow = "show"
        elif show_overflow or show_underflow:
            flow = "hint"
        else:
            flow = "none"

    # Extract axis parameters
    ylabel = kwargs.pop("ylabel", "Density" if density else "Candidates")
    xlabel = kwargs.pop("xlabel", getattr(h, "label", ""))
    ylim_bottom = kwargs.pop("ylim_bottom", 0)
    ylim_top = kwargs.pop("ylim_top", None)
    xlim = kwargs.pop("xlim", None)

    # Extract grid parameters
    grid = kwargs.pop("grid", False)
    grid_alpha = kwargs.pop("grid_alpha", 0.3)

    # Plot histogram using mplhep
    try:
        mplhep.histplot(
            h,
            ax=ax,
            histtype=histtype,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            flow=flow,
            label=label if show_label_in_legend else None,
            density=density,
            **kwargs,  # Pass any remaining kwargs to mplhep.histplot
        )
    except Exception as e:
        raise RuntimeError(f"Error plotting histogram: {e}") from None

    # Configure axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set y-axis limits
    if logy:
        ax.set_yscale("log")
        # For log scale, set a small positive bottom limit
        current_ylim = ax.get_ylim()
        if ylim_bottom <= 0:
            ylim_bottom = max(0.1, current_ylim[0])

    if ylim_top is not None:
        ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
    else:
        ax.set_ylim(bottom=ylim_bottom)

    # Set x-axis limits if specified
    if xlim is not None:
        ax.set_xlim(xlim)

    # Configure grid
    if grid:
        ax.grid(True, alpha=grid_alpha)

    # Add statistics box if requested
    if show_stats:
        _add_stats_box(ax, h, density=density)


def _add_stats_box(ax: plt.Axes, h: Hist, density: bool = False) -> None:
    """Add a statistics text box to the plot."""
    try:
        entries = int(h.sum())

        # Calculate mean and std if possible
        if hasattr(h, "axes") and len(h.axes) > 0:
            # Get bin centers and values
            bin_centers = h.axes[0].centers
            bin_values = h.values()

            if entries > 0:
                # Calculate weighted mean and std
                weights = bin_values / np.sum(bin_values) if np.sum(bin_values) > 0 else bin_values
                mean = np.average(bin_centers, weights=weights)
                variance = np.average((bin_centers - mean) ** 2, weights=weights)
                std = np.sqrt(variance)

                stats_text = f"Entries: {entries}\nMean: {mean:.3f}\nStd: {std:.3f}"
            else:
                stats_text = f"Entries: {entries}\nMean: N/A\nStd: N/A"
        else:
            stats_text = f"Entries: {entries}"

        # Add text box
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    except Exception:
        # If stats calculation fails, just show entries
        try:
            entries = int(h.sum())
            ax.text(
                0.02,
                0.98,
                f"Entries: {entries}",
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        except Exception:
            pass  # If even basic stats fail, don't show anything


# Convenience functions for common plot styles
def plot_hist1d_filled(ax: plt.Axes, h: Hist, **kwargs) -> None:
    """Plot histogram with filled style."""
    kwargs.setdefault("histtype", "stepfilled")
    kwargs.setdefault("alpha", 0.7)
    plot_hist1d(ax, h, **kwargs)


def plot_hist1d_outline(ax: plt.Axes, h: Hist, **kwargs) -> None:
    """Plot histogram with outline style."""
    kwargs.setdefault("histtype", "step")
    kwargs.setdefault("linewidth", 2)
    plot_hist1d(ax, h, **kwargs)


def plot_hist1d_with_errors(ax: plt.Axes, h: Hist, **kwargs) -> None:
    """Plot histogram with error bars."""
    kwargs.setdefault("yerr", True)  # This depends on mplhep support
    plot_hist1d(ax, h, **kwargs)


def plot_hist1d_comparison(
    hists: list[Hist],
    legends: list[Hist],
    ax: plt.Axes,
    histtypes: list[str],
    colors: list[str],
    normalize: bool = True,
    auto_ylim: bool = True,
    ylim_margin: float = 0.05,
    **kwargs,
):
    """
    Plot multiple histograms on the same axes for comparison.

    Args:
        hists: List of histogram objects to plot
        legends: List of legend labels for each histogram
        ax: Matplotlib axes to plot on
        histtypes: List of histogram types for each plot
        colors: List of colors for each histogram
        normalize: Whether to normalize histograms to density (default: True)
        auto_ylim: Automatically set y-axis limits (default: True)
        ylim_margin: Margin factor for y-axis top limit (default: 0.05)
        **kwargs: Additional arguments passed to plot_hist1d

    Raises:
        ValueError: If input lists have different lengths or are empty
        TypeError: If histogram objects are invalid
    """
    input_lists = [hists, legends, histtypes, colors]
    list_names = ["hists", "legends", "histtypes", "colors"]

    if not all(len(lst) == len(hists) for lst in input_lists):
        lengths = [len(lst) for lst in input_lists]
        raise ValueError(
            f"Inputs must have the same length. Got: {dict(zip(list_names, lengths, strict=False))}"
        )

    # Validate histogram objects
    for i, h in enumerate(hists):
        if not hasattr(h, "values") or not hasattr(h, "sum"):
            raise TypeError(f"hists[{i}] is not a valid histogram object")

    max_value = 0
    plotted_hists = []

    for h, legend, histtype, color in zip(hists, legends, histtypes, colors, strict=True):
        try:
            plot_hist1d(
                ax, h, label=legend, histtype=histtype, color=color, density=normalize, **kwargs
            )

            if auto_ylim:
                if normalize and hasattr(h, "density"):
                    current_max = np.max(h.density())
                else:
                    current_max = np.max(h.values())
                max_value = max(max_value, current_max)

            plotted_hists.append(h)

        except Exception as e:
            print(f"Warning: Failed to plot histogram '{legend}': {e}")
            continue

        if not plotted_hists:
            raise ValueError("No valid histograms were plotted. Check input data.")

        if auto_ylim and max_value > 0:
            # Set y-axis limits based on maximum value
            ax.set_ylim(bottom=0, top=max_value * (1 + ylim_margin))
        # max_density = max(max_density, max(h.density()))
        # ax.set_ylim(bottom=0, top=max_density * 1.05)
        # ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.15)

        if any(legends):
            ax.legend(loc="best", fontsize="small", frameon=False)


def multi_hist1d_comparison(hists, legends, histtypes, colors, **kwargs):

    # Check that all inputs have the same length
    assert (
        len(hists) == len(legends) == len(histtypes) == len(colors)
    ), "All inputs must have the same length"

    # Get the optional keyword arguments
    subplot_width = kwargs.get("plot_width", 2)
    subplot_height = kwargs.get("plot_height", 2)

    # Calculate the number of rows and columns
    num_keys = len(hists[0].keys())
    num_cols = int(np.ceil(np.sqrt(num_keys)))
    num_rows = int(np.ceil(num_keys / num_cols))

    # Calculate the figure size
    fig_width = subplot_width * num_cols
    fig_height = subplot_height * num_rows

    # Create the figure and axes
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_width, fig_height))

    # Flatten the axes
    axes = axes.flatten()

    # Loop over the keys and axes
    for key, ax in zip(hists[0].keys(), axes, strict=False):
        h = [h[key] for h in hists]
        plot_hist1d_comparison(h, legends, ax, histtypes, colors, **kwargs)
        # ax.set_title(key)
        # ax.legend()

    # Remove any unused axes
    for ax in axes[num_keys:]:
        ax.remove()

    # Adjust the layout
    plt.tight_layout()
    plt.legend(loc="best")
