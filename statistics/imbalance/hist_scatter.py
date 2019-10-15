import matplotlib.pyplot as plt
import scipy.stats


def hist_scatter(x, y, scatter_kwargs=None, scatter_set_kwargs=None, with_trends=True):
    for v in [scatter_kwargs, scatter_set_kwargs]:
        if v is None:
            v = {}

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(6, 6),
        gridspec_kw=dict(
            width_ratios=[0.7, 0.3], height_ratios=[0.3, 0.7], wspace=0, hspace=0
        ),
    )

    # Bottom left gets the scatter plot
    scatter_ax = ax[1][0]
    scatter_ax.scatter(x, y, **scatter_kwargs)
    scatter_ax.set(**scatter_set_kwargs)
    xlim_orig, ylim_orig = scatter_ax.get_xlim(), scatter_ax.get_ylim()
    xlim = (min(xlim_orig[0], ylim_orig[0]), max(xlim_orig[1], ylim_orig[1]))
    ylim = xlim
    scatter_ax.set(xlim=xlim, ylim=ylim)
    scatter_ax.tick_params(axis="x", which="both", bottom=True, top=False)
    scatter_ax.tick_params(axis="y", which="both", left=True, right=False)
    if with_trends:
        scatter_ax.plot(xlim, ylim, color="black")
        val, edges, _ = scipy.stats.binned_statistic(x, y, "mean", 10)
        scatter_ax.plot((edges[1:] + edges[:-1]) / 2, val, label=r"$\overline{y} | x$")
        val, edges, _ = scipy.stats.binned_statistic(y, x, "mean", 10)
        scatter_ax.plot(val, (edges[1:] + edges[:-1]) / 2, label=r"$\overline{x} | y$")
        scatter_ax.legend()

    # Draw to setup the ticks
    ticks, ticklabels = scatter_ax.get_xticks(), scatter_ax.get_xticklabels()
    fig.canvas.draw()

    # Top left gets the x histogram
    x_hist_ax = ax[0][0]
    x_hist_ax.hist(x)
    x_hist_ax.set(xlim=xlim, xticks=ticks, xticklabels=ticklabels)
    x_hist_ax.tick_params(
        axis="x", which="both", bottom=True, top=True, labelbottom=False, labeltop=True
    )

    # Bottom right get the y hist (on the side)
    y_hist_ax = ax[1][1]
    y_hist_ax.hist(y, orientation="horizontal")
    y_hist_ax.set(ylim=ylim, yticks=ticks, yticklabels=ticklabels)
    y_hist_ax.tick_params(
        axis="y", which="both", left=True, right=True, labelleft=False, labelright=True
    )

    # Top right is empty!
    empty_ax = ax[0][1]
    empty_ax.set_axis_off()

    return fig, ax
