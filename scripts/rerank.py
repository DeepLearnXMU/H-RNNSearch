#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich
# Distributed under MIT license
import argparse
import sys


def main(args):
    k = args.k
    if k is None:
        k = float('inf')

    cur = 0
    best_score = float('inf')
    best_sent = ''
    idx = 0
    for line in sys.stdin:
        num, sent, scores = line.split(' ||| ')

        # new input sentence: print best translation of previous sentence, and reset stats
        if int(num) > cur:
            sys.stderr.write('{} {} \n'.format(cur, best_score))
            sys.stdout.write('{}\n'.format(best_sent))
            cur = int(num)
            best_score = float('inf')
            best_sent = ''
            idx = 0

        # only consider k-best hypotheses
        if idx >= k:
            continue

        scores = map(float, scores.split())
        if args.select_gt:
            scores = scores[args.select_gt:]
        elif args.select:
            scores = [scores[i] for i in args.select]

        score = sum(scores)
        if score < best_score:
            best_score = score
            best_sent = sent.strip()

        idx += 1

    # end of file; print best translation of last sentence
    sys.stderr.write('{} {} \n'.format(cur, best_score))
    sys.stdout.write('{}\n'.format(best_sent))
    # print best_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, help="only consider k-best hypotheses")
    parser.add_argument("--select-gt", type=int, help="select scores indexed from i for reranking")
    parser.add_argument("--select", type=int, nargs='+', help="specify indexed scores for ranking")

    args = parser.parse_args()

