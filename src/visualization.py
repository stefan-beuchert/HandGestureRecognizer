## This file handles all graphical output as the scatter plot of a 2-dim distribution of the data. The classes are
# represented with different colors.

#TODO: Still there has to be written the PCA here to redimension the data from 63 dims to 2

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from data_management import *
from sklearn import decomposition


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
    s = [2 for n in range(len(y_numbered))]

    # make the scatter
    scat = ax.scatter(_X[:,0], _X[:,1], s=s, c=y_numbered, cmap=cmap, norm=norm)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    cb.set_label('Classes')
    ax.set_title('Distribution of classes in x and y direction')
    plt.show()





def plot_pca(_filename: str):

    #<X, y = load_data("data/all_data_preprocessed.csv")
    _, X, _, y = load_data_and_split(_filename)
    #make_scatter_plot(X,y)


    pca = decomposition.PCA(n_components=8)
    pca.fit(X)
    pcs = pca.transform(X)
    print(pca.explained_variance_ratio_)

    make_scatter_plot(pcs[:,:2], y)

plot_pca("data/all_data_preprocessed.csv")

    # Reorder the labels to have colors matching the cluster results
    #y = np.choose(y, [1, 2, 0]).astype(float)
    #ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

    #ax.w_xaxis.set_ticklabels([])
    #ax.w_yaxis.set_ticklabels([])
    #ax.w_zaxis.set_ticklabels([])

    #plt.show()