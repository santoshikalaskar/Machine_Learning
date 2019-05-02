import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt


class Util:

    # function for bar plot

    def bar_plot(self):

        # load required data set
        df = sb.load_dataset('titanic')

        # create bar plot with x and y axis
        sb.barplot(x="sex", y="survived", hue="class", data=df)

        # title for bar plot
        plt.title("BAR PLOT")

        # show bar plot
        plt.show()

    # function for point plot

    def point_plot(self):

        # load required data set
        df = sb.load_dataset('titanic')

        # create point plot with x and y axis
        sb.pointplot(x="sex", y="survived", hue="class", data=df)

        # title for bar plot
        plt.title("POINT PLOT")

        # show point plot
        plt.show()

    # function for scatter plot

    def scatter_plot(self):

        # load required data set/ read csv file
        df_two = pd.read_csv('tips.csv')

        # create scatter plot with x and y axis
        sb.scatterplot(data=df_two, x="day", y="total_bill")

        #  title for scatter plot
        plt.title("SCATTER PLOT")

        # show scatter plot
        plt.show()

    # function for violin float

    def violin_plot(self):

        # load required data set
        df = sb.load_dataset('tips')

        # create violin plot with x and y axis
        sb.violinplot(x="sex", y="total_bill", data=df)

        # title for violin plot
        plt.title("VIOLIN PLOT")

        # show violin plot
        plt.show()

    # function for bao plot

    def box_plot(self):

        # load required data set
        df_one = pd.read_csv('gapminder-FiveYearData.csv')

        # create box plot with x and y axis
        sb.boxplot(data=df_one, x="continent", y="lifeExp")

        # title for box plot
        plt.title("BOX PLOT")

        # show box plot
        plt.show()

    # function for swarm plot

    def swarm_plot(self):

        # load required data set
        df = sb.load_dataset('tips')

        # create swarm plot with x and y axis
        sb.swarmplot(x="day", y="total_bill", data=df)

        # title for swarm plot
        plt.title("SWARM PLOT")

        # show swarm plot
        plt.show()