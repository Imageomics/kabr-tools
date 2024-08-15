import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    data = {"blumenfeld.17": 51,
            "dawson-wabuyele.1": 0,
            "kholim": 352,
            "kline.377": 45,
            "magersupp.2": 71,
            "ninavt": 38,
            "perera.62": 0,
            "rameshbabu.3": 0,
            "ramirez.528": 34,
            "sheets.256": 124,
            "silva.287": 92,
            "sowbaranika": 0,
            "stevens.994": 0,
            "thompson.4509": 0,
            "young.3105": 97}

    nickname2name = {"blumenfeld.17": "Zoe Blumenfeld",
                     "dawson-wabuyele.1": "Kit Dawson-Wabuyele",
                     "kholim": "Maksim Kholiavchenko",
                     "kline.377": "Jenna Kline",
                     "magersupp.2": "Mia Magersupp",
                     "ninavt": "Nina van Tiel",
                     "perera.62": "Rumali Perera",
                     "rameshbabu.3": "Reshma Ramesh Babu",
                     "ramirez.528": "Michelle Ramirez",
                     "sheets.256": "Alec Sheets",
                     "silva.287": "Eduardo Silva",
                     "sowbaranika": "Sowbaranika Balasubramaniam",
                     "stevens.994": "Sam Stevens",
                     "thompson.4509": "Matthew Thompson",
                     "young.3105": "Ainsley Young"}

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
