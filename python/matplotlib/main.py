#!/usr/bin/env python3

"""
Me messing around with matplotlib while reading the tutorial:
    https://matplotlib.org/tutorials/index.html
"""

import matplotlib.pyplot as plt
import numpy as np
from colossus.cosmology import cosmology

my_cosmo_lcdm = {'flat': True, 'H0': 73.0, 'Om0': 0.238, 'Ob0': 0.045714, 'sigma8': 0.8, 'ns': 0.951}
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo_lcdm)

x = np.logspace(10, 15, num=50)
y1 = cosmo.lagrangianR(x)

# Don't do this. It uses a very sneaky api
def just_plot():
    # plt.plot takes a series of ([x], y, [format]) args
    # if x isn't given, uses indexes [0, 1, ..., n]
    # if format isn't given, has some algo to choose
    # https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
    plt.plot(x, y1, 2*y1, 'b-')

    # plt.plot can also take a bunch of kwargs. These can set line parameters
    # Note that the kwargs will applied to ALL of the lines
    # Note also that successive calls to plot adds lines to the same graph
    plt.plot(x, 3* y1, label="Line!")

    # plt.plot returns a list (one for each line) of Line2D objects
    # These have a bunch of set_/get_ methods to change things about the line (color, width, label etc)
    # https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html
    line = plt.plot(x, 4*y1)

    # Each line lives on an axis. These axes are shared between lines
    line2 = plt.plot(x, 5*y1)
    axis = line[0].axes
    assert line[0].axes is line2[0].axes

    # Axis controls a bunch of stuff about the plot
    # https://matplotlib.org/api/axes_api.html
    axis.loglog() # makes it loglog
    axis.legend() # adds the legend (uses line labels)
    axis.set_title("My plot")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    print(axis.get_legend())
    plt.show()

def create_then_plot():
    # An alternative - set up axes first
    # This is the same as fig = plt.figure; ax = fig.add_subplots(111)
    # nrows=1, ncols=1 are the default hence ^
    fig, ax = plt.subplots()
    # fig can be used to add subplots and do stuff with the figure in general
    # change DPI, save it
    # lets you do things across axes (general legend)

    ax.plot(x, y1) # same as plt.plot I think

    # Can use axis to bulk set properties. Just take the set_X function and
    # pass X="" in the kwargs
    ax.set(xscale="log", yscale="log", title="Yolo", xlabel="X")

    # In non-interactive mode, display all figures and block until the figures have been closed
    plt.show(block=False) # yolo no block. Not sure why you would do this...

def create_multi_fig():
    fig, ax = plt.subplots()
    ax.plot(x, y1)
    fig, ax = plt.subplots()
    ax.plot(x, 2*y1)
    plt.show() # shows them both

def create_subplots():
    fig, ax = plt.subplots(nrows=1, ncols=2)
    assert len(ax) == 2
    assert len(fig.get_axes()) == 2 # useful if you lose your axes? Though these aren't the same objs as ax
    ax[0].plot(x, y1)
    ax[1].plot(x, 2*y1)

    plt.show()

def general_best_practice():
    # explicitly create your figure and axes
    # N.B. ax is an array if required, else is just a single axes
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Figure title")

    # Plot on the axes, one line at a time, passing params for the line as kwargs
    ax[0].plot(x, y1, label="Yolo label", linestyle="solid")
    ax[0].scatter(x, 2*y1, label="Yolo label2", color="blue")

    ax[1].plot(x, 3*y1, label="Yolo label", linestyle="solid")

    # Configure the axes with a single "set" command
    # slightly annoyingly, turning the legend on is a second step
    for axis in [ax[0], ax[1]]:
        axis.set(xscale="log", yscale="log", xlabel="X", ylabel="Y")
        axis.legend()

    # If you need to do more custom things it is easy because you now have a framework.
        # A figure, fig, to do things at the highest level
        # An array of axes, ax, which are single plots in the figure
            # Use these to customise anything about the single plot
        # An array of children/lines that are the lines in the figure
        # and all of these things can be accessed/changed
    ax[0].get_lines()[0].set_color("red")
    plt.show()

def plot_multiple_arrays():
    y = np.array(np.array([1,2,3]), np.array([3,2,1]))
    _, ax = plt.subplots()
    ax.plot(y)


if __name__ == "__main__":

    # I think this might be why mpl is confusing. It has these two ways of doing things.
    # just_plot shows the state machine way. You just keep hitting it with commands and it tracks state
        # Yeah, the state machine is fucking confusing. It assumes a bunch of what it thinks you mean.
        # Zen of python - Explicit is better than implicit.
    # create_then_plot shows the OO way. You create a figure object which contains axis objects
    # https://matplotlib.org/tutorials/introductory/lifecycle.html#sphx-glr-tutorials-introductory-lifecycle-py
    # They say use the OO interface and I agree.

    # https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
    # A useful high level summary of what things are and where they live

    # just_plot()
    # create_then_plot()

    # create_multi_fig()
    # create_subplots()

    general_best_practice() # as of what I think now...
