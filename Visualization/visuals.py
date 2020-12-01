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

    def plot_joint(self, scores, names):
        score_dic = {name : score for name, score in zip(names, scores)}
        joint_data = pd.DataFrame(score_dic)
        pair = sns.pairplot(joint_data)
        pair.savefig("pairplot.png")
        pass
        