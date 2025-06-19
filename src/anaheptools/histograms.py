import hist
import matplotlib.pyplot as plt
import mplhep
import numpy as np
from hist import Hist

from .utils import is_collection


def hist_from_array(arr, branches, name, label, range, bins, transform=None, weight=None, **kwargs):
    overflow = kwargs.pop("overflow", False)
    underflow = kwargs.pop("underflow", False)
    h = Hist(
        hist.axis.Regular(
            bins,
            *range,
        ),
        storage=hist.storage.Weight(),
    )

    h.overflow = overflow
    h.underflow = underflow
    h.label = label
    h.name = name

    if transform is not None:
        data = (transform(*(arr[b] for b in branches)),)
    else:
        data = (arr[b] for b in branches)

    h.fill(*data, weight=weight)
    return h


def hist_from_var(var, arr, weight=None):
    """
    Create a histogram from a variable and an input array.
    Variable definition is expected to have the following attributes:
    - branch: name of the branch in the input array
    - name: name of the histogram
    - label: label of the histogram
    - range: range of the histogram
    - bins: number of bins
    - transform: function to transform the input array
    """
    branch = var.branch
    if not is_collection(branch):
        branch = [branch]
    return hist_from_array(
        arr,
        branch,
        var.name,
        var.label,
        (var.x_min, var.x_max),
        var.n_bins,
        var.expression,
        weight=weight,
    )


def plot_hist(ax, h, logy=False, **kwargs):
    histtype = kwargs.get("histtype", "step")
    color = kwargs.get("color", "black")
    label = kwargs.get("label", None)
    flow = kwargs.get("flow", "none")
    density = kwargs.get("density", False)
    alpha = kwargs.get("alpha", 1)
    mplhep.histplot(
        h,
        ax=ax,
        histtype=histtype,
        color=color,
        flow=flow,
        label=label,
        density=density,
        alpha=alpha,
    )
    ax.set_xlabel(h.label)
    ax.set_ylabel("Candidates")
    ax.set_ylim(bottom=0)
    # ax.set_title(h.name)
    if logy:
        ax.set_yscale("log")


def plot_hist1d_comparison(hists, legends, ax, histtypes, colors, **kwargs):
    max_density = 0
    for h, leg, ht, c in zip(hists, legends, histtypes, colors, strict=False):
        plot_hist(ax, h, label=leg, histtype=ht, color=c, density=True, **kwargs)
        max_density = max(max_density, max(h.density()))
    ax.set_ylim(bottom=0, top=max_density * 1.05)
    # ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.15)


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
