import numpy as np
from Scoring.Scores import MoverScore, BLEUScore
from Visualization.visuals import Visualizor

reference_path = "Data/kalm_data/references/subtaskC_gold_answers.csv"
prediction_path = "Data/kalm_data/predictions/subtaskC_answers.csv"
vis = Visualizor()

## -------- Mover Score --------

MS = MoverScore(prediction_path, reference_path)
print("Mover Scores for similarity: ", np.mean(MS.scores))
MS.print_bad_results()

## -------- BLEU Score --------
BS = BLEUScore(prediction_path, reference_path)
BS.print_bad_results()
print("BLEU Scores for similarity: ", np.mean(BS.scores))

## Plotting results

vis.plot_hist(MS.scores, outfile_name="MS_hist")
vis.plot_hist(BS.scores, outfile_name="BS_hist")
vis = Visualizor()
# TODO make scores of same length
# vis.plot_joint(MS.scores, BS.scores)
