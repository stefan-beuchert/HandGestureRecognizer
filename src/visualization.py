## This file handles all graphical output as the scatter plot of a 2-dim distribution of the data. The classes are
# represented with different colors.

#TODO: Still there has to be written the PCA here to redimension the data from 63 dims to 2

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from data_management import *
from sklearn import decomposition
from config import COMBINED_CLASSES


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
    ax.set_title('PCA diagram (49% variance)')
    plt.xlabel("PC1 (30.0%)")
    plt.ylabel("PC2 (19.0%)")
    plt.savefig("figures/PCA.png", dpi=300)
    plt.show()


def plot_pca(_filename: str):

    _, X, _, y = load_data_and_split(_filename)

    # change the y from a letter to an integer number
    y = turn_string_label_to_int(y, COMBINED_CLASSES)

    # do Principal component analysis with 8 components even when just 2 are displayed
    pca = decomposition.PCA(n_components=8)
    pca.fit(X)
    pcs = pca.transform(X)
    print(pca.explained_variance_ratio_)

    make_scatter_plot(pcs[:,:2], y)

def heatmap(_string: str):
    # importing data
    _array = genfromtxt(_string, delimiter=',', dtype=int, skip_header=False)
    _array_for_text = genfromtxt(_string, delimiter=',', dtype=int, skip_header=False)

    # change format (diagonal is "good"), other bad
    np.fill_diagonal(_array, _array.diagonal() * -1)
    _array = _array * -1

    # generate plots
    fig, ax = plt.subplots()
    cmap = "RdYlGn"     # heatmap should have a continuous color green -> yellow -> red
    im = ax.imshow(_array, cmap=cmap, norm=mpl.colors.Normalize(vmin=-400, vmax=400))

    # define axes
    if (_array.shape[0] == 27):
        ticks = ["Class 01", "Class 02", "Class 03", "Class 04", "Class 05", "Class 06", "Class 07", "Class 08", "Class 09",
             "Class 10",
             "Class 11", "Class 12", "Class 13", "Class 14", "Class 15", "Class 16", "Class 17", "Class 18", "Class 19",
             "Class 20",
             "Class 21", "Class 22", "Class 23", "Class 24", "Class 25", "Class 26", "Class 27"]
    else:
        ticks = ["Collection 01 (1-3)", "Collection 02 (4,17)", "Collection 03 (5,6)", "Collection 04 (7,8,20,21)",
                 "Collection 05 (9,13)", "Collection 06 (10,19)",
                 "Collection 07 (11,18)",
                 "Class 12", "Class 14", "Class 15", "Class 16", "Class 22",
                 "Class 23", "Class 24", "Class 25", "Class 26", "Class 27"]

    # Show all ticks and label them with the respective list entries
    plt.xticks(np.arange(_array.shape[0]), labels=ticks, size=4)
    plt.yticks(np.arange(_array.shape[0]), labels=ticks, size=4)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(_array_for_text.shape[0]):
        for j in range(_array_for_text.shape[1]):
            text = ax.text(j, i, _array_for_text[i, j],
                           ha="center", va="center", size=4)

    ax.set_title("Confusion matrix for NN with combined classes")
    fig.tight_layout()

    # save figure and display it
    plt.savefig("figures/NN_heatmap_combined.png", dpi=300)
    plt.show()


#plot_pca("data/all_data_preprocessed.csv")
heatmap("tables/nn_confusion_matrix_combined.csv")
