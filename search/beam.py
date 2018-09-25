# search.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import itertools

import numpy as np


# score: a beam_size * num_vars matrix, represent current score
# n: max number of elements to select
# threshold: prune if score < best + threshold
def find_nbest(score, n, threshold=None):
    num_vars = score.shape[1]

    score = score.flatten()
    nbest = np.argpartition(score, n)[:n]  # get indices of n smallest costs(unordered because of argpartition impl)

    beam_indices = nbest / num_vars
    word_indices = nbest % num_vars
    nbest_score = score[nbest]

    if threshold:  # seems incorrect
        best = np.max(nbest_score)
        cond = nbest_score > best + threshold
        nbest_score = nbest_score[cond]
        beam_indices = beam_indices[cond]
        word_indices = word_indices[cond]

    return nbest_score, beam_indices, word_indices


class beam:
    def __init__(self, beamsize, threshold=None):
        self.size = beamsize
        self.threshold = threshold
        self.scores = []
        self.candidates = []

    def prune(self, log_probs, done_predicate, done_block_predicate, prev_beam, mind, sublen):
        prev_score = np.array(prev_beam.scores, log_probs.dtype)
        score = prev_score[:, None] - log_probs  # nll

        nbest_score, beam_indices, word_indices = find_nbest(score, self.size, self.threshold)

        finished = []
        remained = []
        newmind = []

        for score, bid, wid in itertools.izip(nbest_score, beam_indices, word_indices):
            prev_candidates = prev_beam.candidates
            candidates = prev_candidates[bid] + [wid]

            if not done_block_predicate(candidates) and not done_predicate(candidates):
                #print 'yoyo1',
                remained.append(bid)
                self.candidates.append(candidates)
                self.scores.append(score)
                newmind.append(mind[bid])
            elif done_block_predicate(candidates) and mind[bid] + 1 < sublen:
                #print 'yoyo2',
                remained.append(bid)
                self.candidates.append(candidates)
                self.scores.append(score)
                newmind.append(mind[bid] + 1)
            else:
                #print 'yoyo3',
                finished.append([candidates, score])
            # if done_predicate(candidates):
            #     finished.append([candidates, score])
            # else:
            #     remained.append(bid)
            #     self.candidates.append(candidates)
            #     self.scores.append(score)

        return finished, remained, newmind
