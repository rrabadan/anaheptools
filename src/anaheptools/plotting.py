import os

import matplotlib.pyplot as plt


def plot(name, prefix, figsize=(3.5, 2.7)):
    """
    Auxiliary function to simplify matplotlib plotting
    (using "with" statement). Opens the subplot and
    after yielding saves the figs to .pdf file
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.15, left=0.18, right=0.95, top=0.95)
    yield fig, ax
    fig.savefig(prefix + name + ".pdf")


def save_fig(fig_id, folder, tight_layout=True, fig_extension="jpg", resolution=300):
    path = os.path.join(folder, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


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
