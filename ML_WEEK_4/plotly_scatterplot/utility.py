import matplotlib.pyplot as plt
import pandas as pd
from pylab import randn
import seaborn as sb
import mplcursors
import numpy as np


class Util:

    """draw a scatter plot for random 1000 x and y coordinates"""

    def draw_scatter_plot(self, x_axis, y_axis):
        X = randn(x_axis)
        Y = randn(y_axis)
        plt.scatter(X, Y, color='r')
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title("Scatter Plot with random number")
        plt.show()

    """draw line and scatter plots for random 100 x and y coordinates"""

    def draw_line(self, x_axis, y_axis):

        X = randn(x_axis)
        Y = randn(y_axis)
        # it shows lines for random X and Y values
        plt.scatter(X, Y, color='b')
        plt.plot(X, Y)
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title("Lines with scatter plot")
        plt.show()

    """draw a scatter plot for random 500 x and y coordinates and style it"""

    def sctter_plot_style(self, x_y_axis):
        # Fixing random state for reproducibility
        # np.random.seed(19680801)

        x = np.random.rand(x_y_axis)
        y = np.random.rand(x_y_axis)
        colors = np.random.rand(x_y_axis)
        area = (30 * np.random.rand(x_y_axis)) ** 2
        # 0 to 15 point radii
        plt.scatter(x, y, s=area, c=colors, alpha=0.5)
        # show color scale
        plt.colorbar()
        plt.title("Scatter plot with style")
        plt.show()

    def data_set(self):
        # read data from given csv file
        df = pd.read_csv("data1.csv")
        print(len(df))
        sb.scatterplot(x="total_bill", y="tip",data=df)
        # used to show hover effect,
        # When hover is set to True, annotations are displayed when the mouse hovers over the point
        mplcursors.cursor(hover=True)
        plt.title("Scatter plot for a given dataset and with hover")
        plt.show()
