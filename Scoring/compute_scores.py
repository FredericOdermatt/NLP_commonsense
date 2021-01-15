
import argparse
import numpy as np
import pandas as pd
import sys
import os

'''
from Scores import MeteorScore
from Scores import BLEUScore, RougeScore
from Scores import BertScore, MoverScore
'''

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
reference_path = execution_dir + "/../" + args.ref_path
prediction_path = execution_dir + "/../" + args.pred_path

# In case you only want to consider the 100 sentences that were evaluated from humans this should be set to true. 
# Currently, we only have the human evaluated files for KALM. Hence this can only be used for testing KALMs output so far.
human_evaluated_only = False
csv_dic = {}
out_path = './../data100/machine_eval_results.csv'

## -------- BLEU Score --------
if eval(args.call_BLEU):
    from Scores import BLEUScore
    BS = BLEUScore(prediction_path, reference_path, which="own", human_eval=human_evaluated_only)
    print("BLEU Scores for similarity: ", np.mean(BS.scores))
    csv_dic['BLEU'] = BS.scores
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
    csv_dic['ROUGE'] = RS.scores
    if eval(args.print_bad):
        RS.print_bad_results()


## -------- METEOR Score --------
# does not work with GPU
if eval(args.call_METEOR):
    from Scores import MeteorScore
    # does not work with GPU
    METEORS = MeteorScore(prediction_path, reference_path, human_eval=human_evaluated_only)
    print("METEOR Scores for similarity: ", np.mean(METEORS.scores))
    csv_dic['METEOR'] = METEORS.scores
    if eval(args.print_bad):
        METEORS.print_bad_results()
    
    if os.path.exists(out_path):
        in_df = pd.read_csv(out_path, header=0)
        in_df['METEOR'] = METEORS.scores
        in_df.to_csv(out_path, index=False)
    else:
        csv_dic['METEOR'] = METEORS.scores
        in_df = pd.DataFrame(csv_dic)
        in_df.to_csv(out_path, index=False)


## -------- Mover Score --------
if eval(args.call_MoverScore):
    from Scores import MoverScore
    MS = MoverScore(prediction_path, reference_path, human_eval=human_evaluated_only)
    print("Mover Scores for similarity: ", np.mean(MS.scores))
    csv_dic['MOVER'] = MS.scores
    if eval(args.print_bad):
        MS.print_bad_results()


## -------- BERT Score --------
if eval(args.call_BERTScore):
    # only works with GPU
    from Scores import BertScore
    BERTS = BertScore(prediction_path, reference_path, human_eval=human_evaluated_only)
    print("BERT Scores for similarity: ", np.mean(BERTS.scores))
    csv_dic['BERT'] = BERTS.scores
    if eval(args.print_bad):
        BERTS.print_bad_results()


'''
if os.path.exists(out_path):
    in_df = pd.read_csv(out_path)
    new_df = pd.DataFrame(csv_dic)
    join_df = pd.concat([in_df, new_df], axis=1, join='inner')
    join_df.to_csv(out_path, index=False)
else:
'''
if not eval(args.call_METEOR):
    out_df = pd.DataFrame(csv_dic)
    out_df.to_csv(out_path, index=False)