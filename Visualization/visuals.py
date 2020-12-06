import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def corrfunc(x,y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    # Unicode for lowercase rho (ρ)
    rho = '\u03C1'
    ax.annotate(f'{rho} = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)

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
        pair = sns.pairplot(joint_data, kind="reg")
        pair.map_lower(corrfunc)
        pair.savefig("pairplot.png")
        pass
        