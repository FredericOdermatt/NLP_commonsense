
import argparse
import numpy as np
import sys
import os

# allow it to find Visualization
sys.path.insert(1, os.path.join(sys.path[0], '..'))

'''
from Scores import MeteorScore
from Scores import BLEUScore, RougeScore
from Scores import BertScore, MoverScore
'''
from Visualization.visuals import Visualizor

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ref_path",
    default=None,
    type=str,
)
parser.add_argument(
    "--pred_path",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--call_BLEU",
    default=True,
    type=str,
)
parser.add_argument(
    "--call_ROUGE",
    default=None,
    type=str,
)
parser.add_argument(
    "--call_METEOR",
    default=None,
    type=str,
)
parser.add_argument(
    "--call_MoverScore",
    default=None,
    type=str,
)
parser.add_argument(
    "--call_BERTScore",
    default=None,
    type=str,
)
parser.add_argument(
    "--print_bad",
    default=None,
    type=str,
)

args = parser.parse_args()

execution_dir = os.path.dirname(os.path.abspath(__file__))
reference_path = execution_dir + args.ref_path
prediction_path = execution_dir + args.pred_path

# In case you only want to consider the 100 sentences that were evaluated from humans this should be set to true. 
# Currently, we only have the human evaluated files for KALM. Hence this can only be used for testing KALMs output so far.
human_evaluated_only = False


## -------- BLEU Score --------
if eval(args.call_BLEU):
    from Scores import BLEUScore
    BS = BLEUScore(prediction_path, reference_path, which="own", human_eval=human_evaluated_only)
    print("BLEU Scores for similarity: ", np.mean(BS.scores))
    if eval(args.print_bad):
        BS.print_bad_results()


## -------- Rouge Score --------
if eval(args.call_ROUGE):
    from Scores import RougeScore
    RS = RougeScore(prediction_path, reference_path, human_eval=human_evaluated_only)
    #RS = RougeScore(prediction_path, reference_path, rouge_type = ['rouge2',2]) 
    # possible values for rouge_type[0] = ['rouge1',..., 'rouge9', 'rougeL']
    # possible values for rouge_type[1] = [0,1,2], 0: recall (ROUGE), 1: precision (BLEU), 2: fmeasure
    print("Rouge Scores for similarity: ", np.mean(RS.scores))
    if eval(args.print_bad):
        RS.print_bad_results()


## -------- METEOR Score --------
# does not work with GPU
if eval(args.call_METEOR):
    from Scores import MeteorScore
    # does not work with GPU
    METEORS = MeteorScore(prediction_path, reference_path, human_eval=human_evaluated_only)
    print("METEOR Scores for similarity: ", np.mean(METEORS.scores))
    if eval(args.print_bad):
        METEORS.print_bad_results()


## -------- Mover Score --------
if eval(args.call_MoverScore):
    from Scores import MoverScore
    MS = MoverScore(prediction_path, reference_path, human_eval=human_evaluated_only)
    print("Mover Scores for similarity: ", np.mean(MS.scores))
    if eval(args.print_bad):
        MS.print_bad_results()


## -------- BERT Score --------
if eval(args.call_BERTScore):
    # only works with GPU
    from Scores import BertScore
    BERTS = BertScore(prediction_path, reference_path, human_eval=human_evaluated_only)
    print("BERT Scores for similarity: ", np.mean(BERTS.scores))
    if eval(args.print_bad):
        BERTS.print_bad_results()





'''
## -------- Human Score --------
human_path = execution_dir + "/Data/kalm_data/human_eval/KaLM-Eval.csv"
human_score = pd.read_csv(human_path)
test_ids = human_score["Input.sample_id"].unique()
human = np.empty((len(test_ids), 3))
for i in range(len(test_ids)):
    human[i] = human_score["Answer.reason1"].loc[human_score["Input.sample_id"] == test_ids[i]].to_numpy()
    noise1 = np.random.uniform(-0.15,0.15)
    noise2 = np.random.uniform(-0.15,0.15)
    noise3 = np.random.uniform(-0.15,0.15)
    human[i,0] += noise1
    human[i,1] += noise2
    human[i,2] += noise3

## Plotting results

vis = Visualizor()
vis.plot_hist(MS.scores, outfile_name="MS_hist")
vis.plot_hist(BS.scores, outfile_name="BS_hist")
scores = [MS.scores, BS.scores, RS.scores, human[:,0], human[:,1], human[:,2], np.mean(human, axis=1)]
names = ["Mover Score", "BLEU Score", "Rouge Score", "Human 1", "Human 2", "Human 3", "Human_mean"]
vis.plot_joint(scores=scores, names=names)
'''
