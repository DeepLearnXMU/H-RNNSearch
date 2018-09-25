# plain.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy

__all__ = ["data_length", "convert_data", "convert_index","gets2tmask"]


def data_length(line):
    return len(line.strip().split())


def tokenize(data):
    return data.split()


def to_word_id(data, voc, unk="UNK"):
    newdata = []
    initid = []
    unkid = voc[unk]

    for d in data:
        idlist = [voc[w] if w in voc else unkid for w in d]
        initlist = [1. if w == voc['<e>'] else 0. for w in idlist]
        newdata.append(idlist)
        initid.append(initlist)

    return newdata, initid


def convert_to_array(data, initid, dtype):
    batch = len(data)
    data_len = map(len, data)
    max_len = max(data_len)

    seq = numpy.zeros((max_len, batch), "int32")
    mask = numpy.zeros((max_len, batch), dtype)
    i = numpy.zeros((max_len, batch), dtype)

    for idx, item in enumerate(data):
        seq[:data_len[idx], idx] = item
        mask[:data_len[idx], idx] = 1.0
    for idx, item in enumerate(initid):
        item = numpy.roll(item,1)
        i[:data_len[idx],idx] = item

    return seq, mask,i

def convert_index_to_array(data, dtype):
    batch = len(data)
    data_len = map(len,data)
    max_len = max(data_len)

    index = numpy.zeros((max_len, batch),"int32")
    backindex = numpy.zeros((max_len, batch), "int32")
    m = numpy.zeros((max_len, batch), dtype)

    for idx, item in enumerate(data):
        index[:data_len[idx], idx] = item
        m[:data_len[idx], idx] = 1.0
        backindex[:data_len[idx], idx] = item

    backindex = backindex + 1
    backindex = numpy.roll(backindex,1,axis = 0)
    backindex[0] = 0
    backindex = backindex * m.astype('int32')
    return index, backindex, m

def convert_data(data, voc, unk="UNK", eos="<eos>", dtype="float32"):
    data = [tokenize(item)[:-1] + [eos] for item in data]
    data, initid = to_word_id(data, voc, unk)
    seq, mask, initid = convert_to_array(data, initid, dtype)

    return seq, mask, initid

def convert_index(data,dtype="float32"):
    data = [tokenize(item) for item in data]
    index, backindex, mask = convert_index_to_array(data,dtype)
    return index, backindex, mask

def gets2tmask(xdata, ydata, index, imask, endbid, dtype="float32"):
    xd = numpy.transpose(xdata)
    yd = numpy.transpose(ydata)
    indexd = numpy.transpose(index)
    imaskd = numpy.transpose(imask)
    ylength = yd.shape[1]
    batch = xd.shape[0]
    xlength = xd.shape[1]

    mask = numpy.zeros((ylength, batch, xlength), dtype)
    no = 0
    for x, y, ind, im in zip(xd, yd, indexd, imaskd):
        block = 0
        for yid in numpy.arange(ylength):
            for xid in numpy.arange(xlength):
                if (block == 0):
                    if (xid <= ind[block]):
                        mask[yid][no][xid] = 1.
                    else:
                        mask[yid][no][xid] = 0.
                elif (block < im.sum()):
                    if (xid <= ind[block] and xid > ind[block - 1]):
                        mask[yid][no][xid] = 1.
                    else:
                        mask[yid][no][xid] = 0.
                else:
                    mask[yid][no][xid] = 1.
            if (y[yid] == endbid):
                block += 1
        no += 1
    return mask