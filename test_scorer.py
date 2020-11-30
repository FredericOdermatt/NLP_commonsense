import numpy as np
from Scoring.Scores import MoverScore, BLEUScore, RougeScore
# from Visualization.visuals import Visualizor

reference_path = "Data/kalm_data/references/subtaskC_gold_answers.csv"
prediction_path = "Data/kalm_data/predictions/subtaskC_answers.csv"

## -------- Mover Score --------
'''
MS = MoverScore(prediction_path, reference_path)
print("Mover Scores for similarity: ", np.mean(MS.scores))
MS.print_bad_results()
'''
#vis = Visualizor()
#vis.plot_hist(MS.scores)

'''
## -------- BLEU Score --------
BS = BLEUScore(prediction_path, reference_path)
BS.print_bad_results()
print("BLEU Scores for similarity: ", np.mean(BS.scores))
'''

## -------- Rouge Score --------
RS = RougeScore(prediction_path, reference_path)
#print(RS._scores)
print("Rouge Scores for similarity: ", np.mean(RS.scores))
