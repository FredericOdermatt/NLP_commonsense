import numpy as np
from Scoring.Scores import MoverScore, BLEUScore
from Visualization.visuals import Visualizor

reference_path = "Data/kalm_data/references/subtaskC_gold_answers.csv"
prediction_path = "Data/kalm_data/predictions/subtaskC_answers.csv"

## -------- Mover Score --------
'''
MS = MoverScore(prediction_path, reference_path)
print("Mover Scores for similarity: ", np.mean(MS.scores))
MS.print_bad_results()

vis = Visualizor()
vis.plot_hist(MS.scores)
'''

## -------- BLEU Score --------
# TODO scores not computed correctly , pipeline complete
BS = BLEUScore(prediction_path, reference_path)
#print("All scores: ", BS.scores)
print("BLEU Scores for similarity: ", np.mean(BS.scores))
