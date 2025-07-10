"""
General plotting utilities, styles, and figure management.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

import matplotlib.pyplot as plt

# FIGURE MANAGEMENT UTILITIES


@contextmanager
def figure_context(
    name: str,
    prefix: str = "",
    figsize: tuple[float, float] = (3.5, 2.7),
    save: bool = True,
    format: str = "png",
    dpi: int = 300,
) -> Generator[tuple[plt.Figure, plt.Axes], None, None]:
    """
    Context manager for creating and saving figures.

    Args:
        name: Figure name (used for filename)
        prefix: Prefix for filename
        figsize: Figure size tuple
        save: Whether to save the figure
        format: File format for saving
        dpi: Resolution for raster formats
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.15, left=0.18, right=0.95, top=0.95)

    try:
        yield fig, ax
    finally:
        if save:
            save_figure(fig, name, prefix=prefix, format=format, dpi=dpi)


def save_figure(
    fig: plt.Figure,
    name: str,
    folder: str = ".",
    prefix: str = "",
    format: str = "pdf",
    dpi: int = 300,
    tight_layout: bool = True,
    bbox_inches: str = "tight",
) -> str:
    """
    Save figure with proper formatting.

    Returns:
        Path to saved file
    """
    if tight_layout:
        fig.tight_layout()

    filename = f"{prefix}{name}.{format}"
    path = os.path.join(folder, filename)

    print(f"Saving figure: {path}")
    fig.savefig(path, format=format, dpi=dpi, bbox_inches=bbox_inches)

    return path


# Backward compatibility - keep old function names
def plot(name, prefix, figsize=(3.5, 2.7)):
    """Deprecated: Use figure_context instead."""
    import warnings

    warnings.warn(
        "plot() is deprecated, use figure_context() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return figure_context(name, prefix, figsize)


def save_fig(fig_id, folder, tight_layout=True, fig_extension="jpg", resolution=300):
    """Deprecated: Use save_figure instead."""
    import warnings

    warnings.warn(
        "save_fig() is deprecated, use save_figure() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    fig = plt.gcf()  # Get current figure
    save_figure(
        fig, fig_id, folder, format=fig_extension, dpi=resolution, tight_layout=tight_layout
    )


# LAYOUT AND SUBPLOT UTILITIES


def create_multiplot_layout(
    n_plots: int, figsize_per_plot: tuple[float, float] = (4, 3), max_cols: int = 3
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create optimal subplot layout for comparisons.

    Args:
        n_plots: Number of subplots needed
        figsize_per_plot: Size of each subplot
        max_cols: Maximum columns before wrapping

    Returns:
        Figure and list of axes
    """
    import math

    if n_plots <= max_cols:
        nrows, ncols = 1, n_plots
    else:
        ncols = min(max_cols, math.ceil(math.sqrt(n_plots)))
        nrows = math.ceil(n_plots / ncols)

    figsize = (ncols * figsize_per_plot[0], nrows * figsize_per_plot[1])
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Ensure axes is always a list
    if n_plots == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    return fig, axes[:n_plots]


def get_color_palette(name: str = "default", n_colors: int = 6) -> list[str]:
    """
    Get predefined color palettes.

    Args:
        name: Palette name ("default", "physics", "colorblind", etc.)
        n_colors: Number of colors needed

    Returns:
        List of color strings
    """
    palettes = {
        "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        "physics": ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33"],
        "colorblind": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"],
    }

    palette = palettes.get(name, palettes["default"])

    # Extend palette if needed
    if n_colors > len(palette):
        # Use matplotlib colormap to extend
        import matplotlib.cm as cm

        cmap = cm.get_cmap("tab10")
        palette.extend([cmap(i) for i in range(len(palette), n_colors)])

    return palette[:n_colors]


# STYLE AND LABEL UTILITIES


def add_experiment_label(
    ax: plt.Axes,
    experiment: str = "LHCb",
    energy: str = "13 TeV",
    luminosity: str | None = None,
    location: str = "upper left",
) -> None:
    """Add experiment label to plot."""
    label_parts = [experiment]

    if energy:
        label_parts.append(f"$\\sqrt{{s}} = {energy}$")

    if luminosity:
        label_parts.append(f"$\\mathcal{{L}} = {luminosity}$")

    label = "\n".join(label_parts)

    # Position mapping
    positions = {
        "upper left": (0.05, 0.95),
        "upper right": (0.95, 0.95),
        "lower left": (0.05, 0.05),
        "lower right": (0.95, 0.05),
    }

    x, y = positions.get(location, (0.05, 0.95))
    ha = "left" if "left" in location else "right"
    va = "top" if "upper" in location else "bottom"

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        ha=ha,
        va=va,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def set_lhcb_style(grid=True, size=16, usetex=False):
    """
    Set matplotlib plotting style close to "official" LHCb style
    (serif fonts, tick sizes and location, etc.)
    """
    plt.rc("font", family="serif", size=size)
    plt.rc("text", usetex=usetex)
    plt.rcParams["axes.linewidth"] = 1.3
    plt.rcParams["axes.grid"] = grid
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.axisbelow"] = False
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["ytick.right"] = True


def set_cms_style(grid=False, size=14, usetex=False):
    """Set CMS-like plotting style."""
    plt.rc("font", family="sans-serif", size=size)
    plt.rc("text", usetex=usetex)
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.grid"] = grid
    plt.rcParams["grid.alpha"] = 0.2
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 0.8
    plt.rcParams["ytick.minor.width"] = 0.8
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.fontsize"] = size - 2
    plt.rcParams["axes.labelsize"] = size
    plt.rcParams["axes.titlesize"] = size + 2
