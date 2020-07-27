from matplotlib import pyplot as plt
import numpy as np
import data_utils
from typing import Union
from matplotlib.transforms import BlendedGenericTransform
import os

# %%

# uncomment this to initially set the rc params to plot graphs using latex
# plt.rcParams.update({'text.color': "black",
#                      'axes.labelcolor': "black",
#                      'axes.facecolor': 'white',
#                      'xtick.color': 'black',
#                      'ytick.color': 'black'})
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('font', size=20)


def get_tab10_color(i: int) -> tuple:
    """
    returns
    :param i:
    :return: tuple of one of the 10 colors based on the tableau color scheme
    """
    cmap = plt.get_cmap('tab10')
    return cmap(i % 10)


def get_graduate_color(i: int, n: int) -> tuple:
    """
    returns a graduated color
    """
    cmap = plt.get_cmap('RdBu')
    return cmap(int(i * (255 / n)))

def plot_pd_hist(data) :
    num_elements = len(data)
    i = 0
    j = round(num_elements / 10)
    while (i < num_elements) :
        print("Calculating histogram section for elements "+str(i)+" to "+str(j)+"...")
        data[i:j].hist()
        i = j
        j = round(num_elements / 10) + i
        i += 1
    """
    data.hist()
    """
    plt.xlabel('Trace Number')
    plt.ylabel('Occurance')
    plt.show()
    path = os.getcwd()
    path = os.path.join(path, 'Plots')
    if not (os.path.exists(path)) :
        os.mkdir(path)
    path = os.path.join(path, "Pandas_Histogram_Trace_Information.pdf")
    plt.savefig(path)
    plt.clf()


def plot_hist(data: list, bins: int, x_label: str = None, y_label: str = None, clip_low: float = None,
              clip_high: float = None, legend: list = None, row_width: int = 0.8, edge_color: str = None, title = None):
    """
    Plots a histogram given a list of lists containing the counts for each element. Check that the categorical values
    do not contain any non UTF-8 characters
    :param legend:  List containing Strings
    :param clip_high: upper boundary for clipping the data
    :param clip_low: lower boundary for clipping the data
    :param data: list or list of lists containing the counts to visualize
    :param bins: int representing the number of bins
    :param x_label: str
    :param y_label: str
    :return: None
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if any(isinstance(el, list) for el in data):
        data = [np.clip(x, clip_low, clip_high) for x in data]
        weights = [np.ones_like(x) / float(len(x)) for x in data]
        colors = [get_tab10_color(i) for i in range(len(data))]

    else:
        data = np.clip(data, clip_low, clip_high)
        weights = np.ones_like(data) / float(len(data))
        colors = get_tab10_color(0)

    ax.hist(data, bins=bins, color=colors, rwidth=row_width, weights=weights,
            alpha=0.9, align='mid', label=legend, edgecolor=edge_color)
    ax.grid(which='minor')
    plt.legend()
    
    if (title != None) :
        plt.title(title)
    fig.show()
    if (title != None) :
        path = os.getcwd()
        path = os.path.join(path, 'Plots')
        if not (os.path.exists(path)) :
            os.mkdir(path)
        path = os.path.join(path, title+".pdf")
        plt.savefig(path)

def plot_scatter(x: Union[list, np.ndarray], y: list, x_label: str = None, y_label: str = None, clip_low: float = None,
                 clip_high: float = None, legend=None, title = None):
    """
    Plots a histogram given a list of lists containing the counts for each element
    :param x: list or numpy array
    :param legend: list containing all labels
    :param clip_high: upper boundary for clipping the data
    :param clip_low: lower boundary for clipping the data
    :param y: list of lists containing the counts to visualize
    :param x_label: str
    :param y_label: str
    :return: None
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    y = [np.clip(i, clip_low, clip_high) for i in y]
    colors = [get_tab10_color(i) for i in range(len(y))]
    for i in range(len(y)):
        ax.plot(x, y[i] / sum(y[i]), label=legend[i], color=colors[i], marker="o")
    ax.grid(which='minor')
    plt.legend()

    fig.show()
    if (title != None) :
        path = os.getcwd()
        path = os.path.join(path, 'Plots')
        if not (os.path.exists(path)) :
            os.mkdir(path)
        path = os.path.join(path, title+".pdf")
        plt.savefig(path)


def plot_bar(counts: list, categories: list, x_label: str = None, y_label: str = None, title: str = None, scale: str = None ):
    """
    Plots a bar chart given a list of counts and a list containing the categorical values
    :param legend:  List containing Strings
    :param counts: list containing the counts to visualize
    :param categories: categorical values corresponding to the counts
    :param x_label: str
    :param y_label: str
    :return: None
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if (scale != None) :
        plt.yscale(scale)
    if (title != None) :
        plt.title(title)
    plt.xticks(rotation=90)
    ax.bar(categories, counts, align='center', alpha=0.9, color=get_tab10_color(0))
    fig.tight_layout()
    fig.show()
    if (title != None) :
        path = os.getcwd()
        path = os.path.join(path, 'Plots')
        if not (os.path.exists(path)) :
            os.mkdir(path)
        path = os.path.join(path, title+".pdf")
        plt.savefig(path)

"""
Given a path and a filename, read the losses dA_loss, dB_loss and g_loss and prepare three subplots showing the corresponding loss functions.

Parameters:
    path: directory where to find the file
    file: the file name; path and file are joined automatically within this method
	
Returns:
    Nothing. The results are stored to (src)/Plots/Loss_Evaluation.pdf
"""
def plot_losses(path, file, save_path) :
    filename = os.path.join(path, file)
    dA_loss, dB_loss, g_loss = data_utils.load_results_from_file(filename)
    xs = np.arange(0, len(dA_loss))
    
    # Prepare all ys
    dA_y1 = [dA_loss[i][0] for i in range(len(dA_loss))]
    dA_y2 = [dA_loss[i][1] for i in range(len(dA_loss))]
    
    dB_y1 = [dB_loss[i][0] for i in range(len(dB_loss))]
    dB_y2 = [dB_loss[i][1] for i in range(len(dB_loss))]
    
    g_y1 = [g_loss[i][0] for i in range(len(g_loss))]
    g_y2 = [g_loss[i][1] for i in range(len(g_loss))]
    
    # Plot all losses into individual subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(12,6))
    line1, = ax1.plot(xs, dA_y1, color='r')
    line2, = ax1.plot(xs, dA_y2, color='b')
    
    ax2.plot(xs, dB_y1, color='r')
    ax2.plot(xs, dB_y2, color='b')
    
    ax3.plot(xs, g_y1, color='r')
    ax3.plot(xs, g_y2, color='b')
    
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Training Iteration')
    ax3.set_xlabel('Training Iteration')
    ax1.set_title('Discriminator A')
    ax2.set_title('Discriminator B')
    ax3.set_title('Generator')
    
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    
    fig.legend((line1, line2), ('Discriminator: Loss on fake images\nGenerator: Loss on sketches', 'Discriminator: Loss on real images\nGenerator: Loss on UIs'), loc='upper center', bbox_to_anchor = [0.5, -0.05], bbox_transform = BlendedGenericTransform(fig.transFigure, ax3.transAxes))
    plt.subplots_adjust(bottom = 0.2)

    # Save figure
    if not (os.path.exists(save_path)) :
        print("Created directory "+save_path+" to store loss evaluation plot.")
        os.mkdir(save_path)
    save_path = os.path.join(save_path, 'Loss_Evaluation.pdf')
    plt.savefig(os.path.join(save_path))
    print("Stored loss evaluation plot to file: "+save_path)