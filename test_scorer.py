import numpy as np
from Scoring.Scores import MoverScore, BLEUScore
from Visualization.visuals import Visualizor
import os

execution_dir = os.path.dirname(os.path.abspath(__file__))
print(execution_dir)
reference_path = execution_dir + "/Data/kalm_data/references/subtaskC_gold_answers.csv"
prediction_path = execution_dir + "/Data/kalm_data/predictions/subtaskC_answers.csv"

## -------- Mover Score --------
'''
MS = MoverScore(prediction_path, reference_path)
print("Mover Scores for similarity: ", np.mean(MS.scores))
MS.print_bad_results()
'''
#vis = Visualizor()
#vis.plot_hist(MS.scores)

## -------- BLEU Score --------
BS = BLEUScore(prediction_path, reference_path)
BS.print_bad_results()
print("BLEU Scores for similarity: ", np.mean(BS.scores))
'''

## -------- Rouge Score --------
RS = RougeScore(prediction_path, reference_path)
#print(RS._scores)
print("Rouge Scores for similarity: ", np.mean(RS.scores))
## Plotting results

vis = Visualizor()
vis.plot_hist(MS.scores, outfile_name="MS_hist")
vis.plot_hist(BS.scores, outfile_name="BS_hist")
# TODO make scores of same length
# vis.plot_joint(MS.scores, BS.scores)
