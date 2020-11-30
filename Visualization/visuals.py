import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class Visualizor:
    def __init__(self):
        pass

    def plot_hist(self, scores, outfile_name="hist"):
        hist = sns.displot(scores)
        hist.set(xlabel='Score')
        hist.savefig(outfile_name + ".png")
        pass

    def plot_joint(self, score1, score2):
        joint_data = pd.DataFrame({"score1": score1, "score2": score2})
        pair = sns.pairplot(joint_data)
        pair.set(xlabel='Scores')
        pair.savefig("pairplot.png")
        pass
        