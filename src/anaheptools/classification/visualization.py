from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hist import Hist

from ..histograms import multi_hist1d_comparison


def plot_correlations(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    xlabels: Sequence[str] | None = None,
    fig_size: tuple[float, float] = (5, 5),
    **kwargs: Any,
) -> None:
    """
    Plot a correlation matrix heatmap for the given DataFrame.
    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - columns (list, optional): The columns to include in the correlation matrix. If None, all columns are included. Default is None.
    - transform (callable, optional): A function to transform the data before calculating the correlation matrix. Default is None.
    - xlabels (list, optional): The labels for the x-axis. If None, the column names are used. Default is None.
    - fig_size (tuple, optional): The size of the figure. Default is (5, 5).
    - **kwargs: Additional keyword arguments to be passed to the correlation calculation.
    Returns:
    - None
    """

    data = df.copy()
    if columns is not None:
        data = data[columns]

    if transform is not None:
        data = transform(data)

    corrmat = data.corr(**kwargs)
    ax1 = plt.subplots(ncols=1, figsize=fig_size)[1]
    opts = {"cmap": plt.get_cmap("RdBu"), "vmin": -1, "vmax": +1}
    heatmap1 = ax1.pcolor(corrmat.values, **opts)
    plt.colorbar(heatmap1, ax=ax1)
    ax1.set_title("Correlations")

    if xlabels is None:
        xlabels = list(corrmat.columns)
    for ax in (ax1,):
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_yticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=90, ha="right")
        ax.set_xticklabels(xlabels, minor=False, rotation=90)
        ax.tick_params(axis="x", labelrotation=90)
        # remove gridlines
        ax.grid(False)

    # save_fig(fig_id)
    return None


def plot_signal_background_comparison(
    signal_hists: dict[str, Hist],
    background_hists: dict[str, Hist],
    signal_label: str = "Signal",
    background_label: str = "Background",
    histtypes: list[str] | None = None,
    colors: list[str] | None = None,
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Quick comparison between signal and background histograms.

    Args:
        signal_hists: Dictionary of signal histograms
        background_hists: Dictionary of background histograms
        signal_label: Label for signal histograms (default: "Signal")
        background_label: Label for background histograms (default: "Background")
        histtypes: List of histogram types (default: ["step", "stepfilled"])
        colors: List of colors (default: ["red", "blue"])
        **kwargs: Additional arguments passed to multi_hist1d_comparison
    """
    if histtypes is None:
        histtypes = ["step", "stepfilled"]
    if colors is None:
        colors = ["red", "blue"]

    return multi_hist1d_comparison(
        hists=[signal_hists, background_hists],
        legends=[signal_label, background_label],
        histtypes=histtypes,
        colors=colors,
        **kwargs,
    )
