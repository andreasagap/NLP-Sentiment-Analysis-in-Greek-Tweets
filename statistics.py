import pandas as pd
import matplotlib.pyplot as plt



if __name__ == '__main__':
    values = [0,0,0,0]
    csv = pd.read_csv("twitter.csv")
    first600 = csv.head(600)
    for index, row in first600.iterrows():
        print(str(row[3]))
        if str(row[3]) != "nan":
            values[int(row[3]) - 1] += 1

    fig = plt.figure()
    plt.title("April")
    categories = ["neutral","optimistic","pessimistic1","pessimistic2"]
    plt.bar(categories, values)
    plt.show()
