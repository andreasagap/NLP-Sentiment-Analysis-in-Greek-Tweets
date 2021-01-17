import pandas as pd
import matplotlib.pyplot as plt
import calendar
from datetime import datetime as dt

def plotBarByMonth():
    m = dt.today().month
    months = dict.fromkeys(calendar.month_name[m - 8:m + 1], [0, 0, 0, 0])
    csv = pd.read_csv("dataset/twitter.csv")
    for index, row in csv.iterrows():
        if str(row[2]) != "nan":
            date_str = str(row[3]).split(" ")[0]
            month = dt.strptime(date_str, '%Y-%m-%d').strftime("%B")
            months[month] = [item + 1 if (int(row[2]) - 1) == i else item for i, item in enumerate(months[month])]

    fig, axs = plt.subplots(1, 3, figsize=(8, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()
    i = 0
    categories = ["neutral", "optimistic", "pessimistic1", "pessimistic2"]
    for key, value in months.items():
        axs[i].set_title(key)
        axs[i].bar(categories, value)
        axs[i].set_yticks([])
        rects = axs[i].patches
        for rect, label in zip(rects, value):
            height = rect.get_height()
            axs[i].text(rect.get_x() + rect.get_width() / 2, height / 2, "N=" + str(int(label*100/sum(value))) + "%",
                        ha='center', va='bottom')
        i += 1
        if (i % 3 == 0):
            plt.show()
            i = 0
            fig, axs = plt.subplots(1, 3, figsize=(8, 6), facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace=.5, wspace=.001)
            axs = axs.ravel()

def plotLine(category,title):
    m = dt.today().month
    months = dict.fromkeys(calendar.month_name[m - 8:m + 1], [0, 0, 0, 0])
    csv = pd.read_csv("dataset/twitter.csv")
    for index, row in csv.iterrows():
        if str(row[2]) != "nan":
            date_str = str(row[3]).split(" ")[0]
            month = dt.strptime(date_str, '%Y-%m-%d').strftime("%B")
            months[month] = [item + 1 if (int(row[2]) - 1) == i else item for i, item in enumerate(months[month])]

    pointsN,pointsO,pointsP1,pointsP2 = [],[],[],[]
    monthx = []
    for key, value in months.items():
        monthx.append(key)
        pointsN.append(value[0])
        pointsO.append(value[1])
        pointsP1.append(value[2])
        pointsP2.append(value[3])

        # pointsN.append(int(value[0] * 100 / sum(value)))
        # pointsO.append(int(value[1] * 100 / sum(value)))
        # pointsP1.append(int(value[2] * 100 / sum(value)))
        # pointsP2.append(int(value[3] * 100 / sum(value)))
    plt.title(title)
    plt.plot(monthx,pointsN, linestyle='--', marker='o', color='b',label="Neutral")
    plt.plot(monthx, pointsO, linestyle='--', marker='o', color='g',label="Optimistic")
    plt.plot(monthx, pointsP1, linestyle='--', marker='o', color='y',label="Pessimistic1")
    plt.plot(monthx, pointsP2, linestyle='--', marker='o', color='r',label="Pessimistic2")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    #plotBarByMonth()
    plotLine(3,"Sentiments in year")
