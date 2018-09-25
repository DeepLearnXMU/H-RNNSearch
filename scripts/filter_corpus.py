import argparse
from collections import defaultdict
import cPickle
import hashlib
import sys
import itertools


def main(args):
    corpus = args.corpus
    f_vocabs = args.vocabs
    unk = args.unk
    ratio = args.ratio
    dedup = args.dedup
    min_length = args.min_length
    max_length = args.max_length

    if min_length is None:
        min_length = 0
    if max_length is None:
        max_length = sys.maxint

    if f_vocabs:
        vocabs = [cPickle.load(open(f, 'rb')) for f in f_vocabs]
    else:
        vocabs = [None] * len(corpus)

    assert len(corpus) == len(vocabs)

    fds = [open(f) for f in corpus]

    hashes = set()

    writers = [open('%s.filtered' % f, 'w') for f in corpus]

    record_dup = 0
    record_ratio = 0  # sentences exceeding unk ratio
    record_length = 0
    while True:
        try:
            lines = [fd.next() for fd in fds]
        except StopIteration:
            break

        newlines = []
        delete = False

        for i, line in enumerate(lines):
            vocab = vocabs[i]
            words = line.split()
            n_l = len(words)
            if vocab:
                words2 = [w if w in vocabs[i] else unk for w in words]
                ratio_l = words2.count(unk) / (n_l + 0.0)

                if ratio_l >= ratio:
                    delete = True
                    record_ratio += 1
                    break
            if n_l < min_length or n_l > max_length:
                delete = True
                record_length += 1
                break

            newlines.append(line)

        if dedup:
            hash_val = hashlib.md5('||||'.join(newlines)).hexdigest()
            if hash_val in hashes:
                delete = True
                record_dup += 1
            hashes.add(hash_val)
        if not delete:
            for writer, line in itertools.izip(writers, newlines):
                writer.write('%s' % line)

    for fd in fds + writers:
        fd.close()

    print 'delete lines exceeding ratio(=%.2f) >= : %d' % (ratio, record_ratio)
    print 'delete duplicates: %d' % record_dup
    print 'delete out-of-range: %d' % record_length


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=str, nargs='+')
    parser.add_argument('--min-length', type=int)
    parser.add_argument('--max-length', type=int)
    parser.add_argument('--vocabs', type=str, nargs='+')
    parser.add_argument('--unk', type=str, default='UNK')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='discard sentence pairs with unk ratio greater or equal than given value')
    parser.add_argument('--dedup', action='store_true', help='remove duplicates')
    parser.add_argument('--replace', action='store_true', help='replace oov word with unk symbol')

    args = parser.parse_args()
    main(args)

