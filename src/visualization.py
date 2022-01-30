## This file handles all graphical output as the scatter plot of a 2-dim distribution of the data. The classes are
# represented with different colors.

#TODO: Still there has to be written the PCA here to redimension the data from 63 dims to 2

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from data_management import *



def make_scatter_plot(_X: np.ndarray, _y: np.ndarray):

    keys = np.unique(_y)
    values = range(len(keys))
    label_conv = dict(zip(keys, values))

    # convert colors to be has
    y_numbered = []
    for value in _y:
        y_numbered.append(label_conv[value])


    N = len(keys) # Number of labels

    # setup the plot
    fig, ax = plt.subplots(1,1, figsize=(6,6))

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # change dot-size
    s = [2 for n in range(len(y))]

    # make the scatter
    scat = ax.scatter(_X[:,0], _X[:,1], s=s, c=y_numbered, cmap=cmap, norm=norm)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    cb.set_label('Classes')
    ax.set_title('Distribution of classes in x and y direction')
    plt.show()

#<X, y = load_data("data/all_data_preprocessed.csv")
_, X, _, y = load_data_and_split("data/all_data_preprocessed.csv")
make_scatter_plot(X,y)
