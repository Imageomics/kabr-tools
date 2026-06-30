import os
import sys
from lxml import etree
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python statistics.py path_to_mini_scenes")
        exit(0)
    elif len(sys.argv) == 2:
        path_to_mini_scenes = sys.argv[1]

    data = {}
    data["Zebra"] = {"Walk": 0,
                     "Graze": 0,
                     "Head Up": 0,
                     "Auto-Groom": 0,
                     "Mutual Grooming": 0,
                     "Trotting": 0,
                     "Running": 0,
                     "Drinking": 0,
                     "Herding": 0,
                     "Lying Down": 0,
                     "Mounting-Mating": 0,
                     "Sniff": 0,
                     "Urinating": 0,
                     "Defecating": 0,
                     "Dusting": 0,
                     "Fighting": 0,
                     "Chasing": 0,
                     "Occluded": 0,
                     "Out of Focus": 0,
                     "Out of Frame": 0}

    data["Giraffe"] = {"Walking": 0,
                       "Head Up": 0,
                       "Auto-Groom": 0,
                       "Mutual Grooming": 0,
                       "Running": 0,
                       "Browsing": 0,
                       "Mounting": 0,
                       "Urinating": 0,
                       "Defecating": 0,
                       "Fighting": 0,
                       "Occluded": 0,
                       "Out of Focus": 0,
                       "Out of Frame": 0}

    data["Baboon"] = {"Walk": 0,
                      "Head Up": 0,
                      "Forage": 0,
                      "Auto-Groom": 0,
                      "Mutual Grooming": 0,
                      "Running": 0,
                      "Infant Carry": 0,
                      "Fighting": 0,
                      "Occluded": 0,
                      "Out of Focus": 0,
                      "Out of Frame": 0}

    for folder in os.listdir(path_to_mini_scenes):
        for file in os.listdir(f"{path_to_mini_scenes}/{folder}/actions"):
            actions_xml = f"{path_to_mini_scenes}/{folder}/actions/{file}"

            if os.path.splitext(file)[1] == ".xml":
                root = etree.parse(actions_xml).getroot()

                for track in root.iterfind("track"):
                    label = track.attrib["label"]

                    for entry in track.iter("points"):
                        outside = entry.attrib["outside"]

                        if outside == "1":
                            continue

                        behavior = "".join(entry.find("attribute").itertext())
                        data[label][behavior] += 1

    sns.set(font_scale=1.2)
    plt.figure(figsize=(17, 17))
    barplot = sns.barplot(x=list(data["Zebra"].keys()), y=[float(data["Zebra"][key]) for key in data["Zebra"].keys()])

    for item in barplot.get_xticklabels():
        item.set_rotation(45)

    plt.savefig("../statistics_zebras.png")
    plt.close()
    plt.figure(figsize=(17, 17))
    barplot = sns.barplot(x=list(data["Giraffe"].keys()),
                          y=[float(data["Giraffe"][key]) for key in data["Giraffe"].keys()])

    for item in barplot.get_xticklabels():
        item.set_rotation(45)

    plt.savefig("../statistics_giraffes.png")
    plt.close()
