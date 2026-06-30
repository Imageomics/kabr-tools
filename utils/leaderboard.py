import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    data = {"annotator_01": 51,
            "annotator_02": 0,
            "annotator_03": 352,
            "annotator_04": 45,
            "annotator_05": 71,
            "annotator_06": 38,
            "annotator_07": 0,
            "annotator_08": 0,
            "annotator_09": 34,
            "annotator_10": 124,
            "annotator_11": 92,
            "annotator_12": 0,
            "annotator_13": 0,
            "annotator_14": 0,
            "annotator_15": 97}

    nickname2name = {"annotator_01": "Annotator 1",
                     "annotator_02": "Annotator 2",
                     "annotator_03": "Annotator 3",
                     "annotator_04": "Annotator 4",
                     "annotator_05": "Annotator 5",
                     "annotator_06": "Annotator 6",
                     "annotator_07": "Annotator 7",
                     "annotator_08": "Annotator 8",
                     "annotator_09": "Annotator 9",
                     "annotator_10": "Annotator 10",
                     "annotator_11": "Annotator 11",
                     "annotator_12": "Annotator 12",
                     "annotator_13": "Annotator 13",
                     "annotator_14": "Annotator 14",
                     "annotator_15": "Annotator 15"}

    for key in list(data.keys()):
        if data[key] == 0:
            del data[key]

    sns.set(font_scale=1.2)
    plt.figure(figsize=(17, 17))
    barplot = sns.barplot(x=[nickname2name[key] for key in data.keys()], y=[float(data[key]) for key in data.keys()])
    barplot.set(ylabel="Number of Annotated Mini-Scenes")

    for item in barplot.get_xticklabels():
        item.set_rotation(45)

    plt.savefig("../leaderboard.png")
    plt.close()
    plt.figure(figsize=(17, 17))
