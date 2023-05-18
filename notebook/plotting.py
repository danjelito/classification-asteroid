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