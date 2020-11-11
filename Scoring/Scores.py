from moverscore_v2 import get_idf_dict, word_mover_score 
from collections import defaultdict


class Scorer:
    def __init__(self):
        self.b = 0

    def MoverScore(self, gen_reason, references):
        # Beispiel Sätze
        # gen_reason = ["A rabbit can not fly because he has no wings.", "The rabbit is running over the moon.","Having breakfast is genious since cheese is delicous."]
        # references = ["The rabbit is not a bird.","Showing mercy is not an option.", "The dinner was very good because the meat was very tender."]

        idf_dict_hyp = get_idf_dict(gen_reason)
        idf_dict_ref = get_idf_dict(references)
        scores = word_mover_score(references, gen_reason, idf_dict_ref, idf_dict_hyp, stop_words=["."], n_gram=1, remove_subwords=True)

        return scores

    def BLEUscore(self):
        a = 0
        