import os
import pandas as pd
import numpy as np
from collections import defaultdict
from Scoring.BLEUScore import own_bleu_score, challenge_score
from Visualization.visuals import Visualizor



class Scorer:
    def __init__(self, prediction_path, reference_path):
        self.prediction_path = prediction_path
        self.reference_path = reference_path
        self.load_data(prediction_path, reference_path)
        self._scores = []

    def load_data(self, prediction_path, reference_path):
        if  not os.path.isfile(prediction_path):
            raise ImportError ('File '+ prediction_path +' does not exist.')
        elif not os.path.isfile(reference_path):
            raise ImportError ('File '+ reference_path +' does not exist.')

        self.predictions_df = pd.read_csv(prediction_path, index_col = 0)
        self.references_df = pd.read_csv(reference_path, index_col = 0)

        if len(self.predictions_df) != len(self.references_df):
            raise ValueError("Number of reference and generated reasons do not match.")

        if any(self.predictions_df.index.to_numpy() != self.references_df.index.to_numpy()):
            raise ValueError("Indices of provided predictions and reference reasons do not match.")

    def compute_scores(self):
        pass

    def preprocess(self):
        pass

    def print_bad_results(self):
        pass

    @property
    def scores(self):
        return self._scores 


    
class BLEUScore(Scorer):
    def __init__(self, prediction_path, reference_path):
        super().__init__(prediction_path, reference_path)
        self.compute_scores()

    def compute_scores(self):
        predictions, references = self.preprocess()
        
        c = 0

        for prediction, reference_triplet in zip(predictions, references):
            #print(prediction, " dann ", reference_triplet)
            reference_triplet = reference_triplet.reshape(3,-1)
            #if c < 3:
            #    print(reference_triplet)
            #c += 1
            self._scores.append(own_bleu_score(prediction, reference_triplet))

    def preprocess(self):
        predictions = self.predictions_df.values
        references = self.references_df.values

        return (predictions, references)

    def print_bad_results(self, shown_elems=3):
        idx = np.argpartition(bleu_scores, shown_elems)
        for i in idx[:shown_elems]:
            print("BLEU Reference:     ", references[i], "\n")
            print("BLEU Answer:     : ", predictions[i], "\n")
            print("BLEU Row Index:     : ", references_df.index[int(i/3.)], "\n")
            print("------------------------------------\n")



class MoverScore(Scorer):

    def __init__(self, prediction_path, reference_path):
        super().__init__(prediction_path, reference_path)
        self.compute_scores()

    def compute_scores(self):
        from moverscore_v2 import get_idf_dict, word_mover_score, plot_example
        from collections import defaultdict
        # only import these module if object instantiated
        # demands GPU 

        # Beispiel Sätze
        # predictions = ["A rabbit can not fly because he has no wings.", "The rabbit is running over the moon.","Having breakfast is genious since cheese is delicous."]
        # references = ["The rabbit is not a bird.","Showing mercy is not an option.", "The dinner was very good because the meat was very tender."]
        predictions, references = self.preprocess()

        idf_dict_hyp = get_idf_dict(predictions)
        idf_dict_ref = get_idf_dict(references)
        self._scores = word_mover_score(references, predictions, idf_dict_ref, idf_dict_hyp, stop_words=["."], n_gram=1, remove_subwords=True)

    def preprocess(self):
        predictions_array = pd.concat([self.predictions_df, self.predictions_df, self.predictions_df], axis=1).to_numpy()
        references_array = self.references_df.to_numpy()
        predictions = predictions_array.reshape((np.prod(predictions_array.shape),)).tolist()
        references = references_array.reshape((np.prod(predictions_array.shape),)).tolist()

        return (predictions, references)

    def print_bad_results(self, shown_elems=3):
        idx = np.argpartition(mover_scores, shown_elems)
        for i in idx[:shown_elems]:
            print("Mover Reference:     ", references[i], "\n")
            print("Mover Answer:     : ", predictions[i], "\n")
            print("Mover Row Index:     : ", references_df.index[int(i/3.)], "\n")
            print("------------------------------------\n") 
        