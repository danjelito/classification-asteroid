import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np


# for plotting num feature
def create_kdeplot(data, ax, x, hue= None):
    sns.kdeplot(
        data= data, 
        x= x, 
        fill= False, 
        hue= hue,
        palette= 'Set1',
        ax= ax,
    )

# for plotting num feature with hue
def create_violinplot(data, ax, x, y):
    sns.violinplot(
        data= data, 
        x= x,
        y= y, # this is equal to hue
        palette= 'Set1', 
        ax= ax
    )

def create_corr_heatmap(data: pd.DataFrame):
    """Create a heatmap showing the correlation between features

    Arguments:
        data -- a DF containing the correlation
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    mask = np.triu(np.ones_like(data, dtype = bool))
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    sns.heatmap(
        data = data, 
        mask = mask, 
        annot= True, 
        annot_kws= {'fontsize': 6},
        fmt= ".2f", 
        cmap= cmap, 
        linewidths= 1,
        vmin=-1, vmax=1, 
        cbar= False
    )

    yticks = [i.title() for i in data.index]
    xticks = [i.title() for i in data.columns]
    plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize= 8)
    plt.xticks(plt.xticks()[0], labels=xticks, fontsize = 8)
    plt.title("Correlation Heatmap", fontsize=18)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='Test set')