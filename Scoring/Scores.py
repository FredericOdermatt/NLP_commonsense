import os
import pandas as pd
import numpy as np

# BLEU Score
from Scoring.BLEUScore import own_bleu_score, challenge_score

# Rouge Score
from rouge_score import rouge_scorer


class Scorer:
    def __init__(self, prediction_path, reference_path):
        self.prediction_path = prediction_path
        self.reference_path = reference_path
        self._references = []
        self._predictions = []
        self._scores = []
        self.predictions_df = pd.DataFrame()
        self.references_df = pd.DataFrame()
        self.load_data(prediction_path, reference_path)


    def load_data(self, prediction_path, reference_path):
        if  not os.path.isfile(prediction_path):
            raise ImportError ('File '+ prediction_path +' does not exist.')
        elif not os.path.isfile(reference_path):
            raise ImportError ('File '+ reference_path +' does not exist.')

        self.predictions_df = pd.read_csv(prediction_path, index_col = 0, names=['out'])
        self.references_df = pd.read_csv(reference_path, index_col = 0, names=['ref1','ref2','ref3'])

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
        # reducing length of floats otherwise memory errors occur
        return list(map(lambda a: round(a,5), self._scores))


    
class BLEUScore(Scorer):
    def __init__(self, prediction_path, reference_path, which="own"):
        self.which = which
        super().__init__(prediction_path, reference_path)
        self.compute_scores()

    def compute_scores(self):
        self.preprocess()

        if self.which == "challenge":
            self._scores.append(self.compute_challenge_score())
        else:
            for prediction, reference_triplet in zip(self._predictions, self._references):
                self._scores.append(own_bleu_score(predictions=prediction, references=reference_triplet))

    def compute_challenge_score(self):
        scores, precisions = challenge_score(self.reference_path, self.prediction_path)
        return scores

    def preprocess(self):
        self._predictions = self.predictions_df.values
        self._references = self.references_df.values

    def print_bad_results(self, shown_elems=3):
        idx = np.argpartition(self._scores, shown_elems)
        for i in idx[:shown_elems]:
            print("BLEU References:     ", self._references[i], "\n")
            print("BLEU Answer:     : ", self._predictions[i], "\n")
            print("BLEU Row Index:     : ", self.references_df.index[int(i/3.)], "\n")
            print("------------------------------------\n")


class MoverScore(Scorer):

    def __init__(self, prediction_path, reference_path):
        super().__init__(prediction_path, reference_path)

        self.compute_scores()

    def compute_scores(self):
        # only import these module if object instantiated
        from moverscore_v2 import get_idf_dict, word_mover_score, plot_example
        from collections import defaultdict
        # demands GPU 

        # Beispiel Sätze
        # predictions = ["A rabbit can not fly because he has no wings.", "The rabbit is running over the moon.","Having breakfast is genious since cheese is delicous."]
        # references = ["The rabbit is not a bird.","Showing mercy is not an option.", "The dinner was very good because the meat was very tender."]
        self.preprocess()
        idf_dict_hyp = get_idf_dict(self._predictions)
        idf_dict_ref = get_idf_dict(self._references)
        all_scores = word_mover_score(self._references, self._predictions, idf_dict_ref, idf_dict_hyp, stop_words=["."], n_gram=4, remove_subwords=True)
        all_scores = np.array(all_scores).reshape((-1,3))
        self._scores = np.max(all_scores, axis=1)

    def preprocess(self):
        predictions_array = pd.concat([self.predictions_df, self.predictions_df, self.predictions_df], axis=1).to_numpy()
        references_array = self.references_df.to_numpy()
        # self._predictions = predictions_array.reshape((np.prod(predictions_array.shape),)).tolist()
        self._predictions = predictions_array.flatten()
        self._references = references_array.flatten()
        #self._references = references_array.reshape((np.prod(predictions_array.shape),)).tolist()

    def print_bad_results(self, shown_elems=3):
        idx = np.argpartition(self._scores, shown_elems)
        for i in idx[:shown_elems]:
            print("Mover Reference:     ", self._references[i], "\n")
            print("Mover Answer:     : ", self._predictions[i], "\n")
            print("Mover Row Index:     : ", self.references_df.index[int(i/3.)], "\n")
            print("------------------------------------\n") 


class RougeScore(Scorer):
    def __init__(self, prediction_path, reference_path, rouge_type = ['rouge2',2]):
        super().__init__(prediction_path, reference_path)
        self.rouge_type = rouge_type
        self.compute_scores()
        
        
    def compute_scores(self):
        self.preprocess()
        
        scorer = rouge_scorer.RougeScorer([self.rouge_type[0]], use_stemmer=True)
        self._scores = list(map(lambda p,x,y,z: max(scorer.score(p,x)[self.rouge_type[0]][self.rouge_type[1]], 
                                                    scorer.score(p,y)[self.rouge_type[0]][self.rouge_type[1]],
                                                    scorer.score(p,z)[self.rouge_type[0]][self.rouge_type[1]]), 
                                                    self._predictions, self._references[0], self._references[1], self._references[2]))

    def preprocess(self):
        predictions_array = self.predictions_df.to_numpy()
        self._predictions = predictions_array.reshape((np.prod(predictions_array.shape),)).tolist()

        references_array = self.references_df.to_numpy()
        self._references = references_array.T.tolist()
    
    def print_bad_results(self, shown_elems=3):
        
        idx = np.argpartition(self._scores, shown_elems)

        for i in idx[:shown_elems]:
            print("References:     ", self._references[0][i], ", ", self._references[1][i],", ", self._references[2][i], "\n")
            print("Answer:     : ", self._predictions[i], "\n")
            print("ROUGE Score:     : ", self._scores[i], ",", self.rouge_type, "\n")
            print("Row Index:     : ", self.references_df.index[int(i)], "\n")
            print("------------------------------------\n")

                
