import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python statistics.py annotations")
        exit(0)
    elif len(sys.argv) == 2:
        annotations = sys.argv[1]

    label2number = {"Walk": 0,
                    "Graze": 1,
                    "Browse": 2,
                    "Head Up": 3,
                    "Auto-Groom": 4,
                    "Trot": 5,
                    "Run": 6,
                    "Occluded": 7}

    number2label = {value: key for key, value in label2number.items()}

    data_all = {"Walk": 0,
                "Graze": 0,
                "Browse": 0,
                "Head Up": 0,
                "Auto-Groom": 0,
                "Trot": 0,
                "Run": 0,
                "Occluded": 0}

    data_g = {"Walk": 0,
              "Graze": 0,
              "Browse": 0,
              "Head Up": 0,
              "Auto-Groom": 0,
              "Trot": 0,
              "Run": 0,
              "Occluded": 0}

    data_zg = {"Walk": 0,
               "Graze": 0,
               "Browse": 0,
               "Head Up": 0,
               "Auto-Groom": 0,
               "Trot": 0,
               "Run": 0,
               "Occluded": 0}

    data_zp = {"Walk": 0,
               "Graze": 0,
               "Browse": 0,
               "Head Up": 0,
               "Auto-Groom": 0,
               "Trot": 0,
               "Run": 0,
               "Occluded": 0}

    df = pd.read_csv(annotations, sep=" ")

    for index, row in df.iterrows():
        data_all[number2label[row["labels"]]] += 1

        if row["original_vido_id"].startswith("G"):
            data_g[number2label[row["labels"]]] += 1
        elif row["original_vido_id"].startswith("ZG"):
            data_zg[number2label[row["labels"]]] += 1
        elif row["original_vido_id"].startswith("ZP"):
            data_zp[number2label[row["labels"]]] += 1

    sns.set(font_scale=1.2)
    plt.figure(figsize=(17, 17))
    barplot = sns.barplot(x=list(data_all.keys()), y=[float(data_all[key]) for key in data_all.keys()])

    for item in barplot.get_xticklabels():
        item.set_rotation(45)

    plt.savefig("../statistics_all.png")
    plt.close()
    plt.figure(figsize=(17, 17))
    barplot = sns.barplot(x=list(data_g.keys()),
                          y=[float(data_g[key]) for key in data_g.keys()])

    for item in barplot.get_xticklabels():
        item.set_rotation(45)

    plt.savefig("../statistics_g.png")
    plt.close()
    plt.figure(figsize=(17, 17))
    barplot = sns.barplot(x=list(data_zg.keys()),
                          y=[float(data_zg[key]) for key in data_zg.keys()])

    for item in barplot.get_xticklabels():
        item.set_rotation(45)

    plt.savefig("../statistics_zg.png")
    plt.close()
    plt.figure(figsize=(17, 17))
    barplot = sns.barplot(x=list(data_zp.keys()),
                          y=[float(data_zp[key]) for key in data_zp.keys()])

    for item in barplot.get_xticklabels():
        item.set_rotation(45)

    plt.savefig("../statistics_zp.png")
    plt.close()
