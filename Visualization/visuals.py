import seaborn as sns
import pandas as pd

class Visualizor:
    def __init__(self):
        pass

    def plot_hist(self, scores):
        hist = sns.displot(scores)
        hist.set(xlabel='Mover Score')
        hist.savefig("score_hist.png")

        pass

    def plot_joint(self, score1, score2):
        joint_data = pd.DataFrame({"score1": score1, "score2": score2})
        hist = sns.pairplot(joint_data)
        hist.set(xlabel='Scores')
        hist.savefig("pairplot.png")

        pass
        