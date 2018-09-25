# rnnsearch.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano
import theano.sandbox.rng_mrg
import theano.tensor as T

import nn
import ops
from bridge import map_key
from encoder import Encoder
from block_encoder import Block_Encoder
from decoder import DecoderGruSimple, DecoderGruCond
from search import beam, select_nbest


class rnnsearch:
    def __init__(self, **option):
        # source and target embedding dim
        sedim, tedim = option["embdim"]
        # source, target and attention hidden dim
        shdim, bshdim, thdim, ahdim, bahdim = option["hidden"]
        # maxout hidden dim
        maxdim = option["maxhid"]
        # maxout part
        maxpart = option["maxpart"]
        # deepout hidden dim
        deephid = option["deephid"]
        svocab, tvocab = option["vocabulary"]
        sw2id, sid2w = svocab
        tw2id, tid2w = tvocab
        # source and target vocabulary size
        svsize, tvsize = len(sid2w), len(tid2w)

        if "scope" not in option or option["scope"] is None:
            option["scope"] = "rnnsearch"

        if "initializer" not in option:
            option["initializer"] = None

        if "regularizer" not in option:
            option["regularizer"] = None

        if "keep_prob" not in option:
            option["keep_prob"] = 1.0

        dtype = theano.config.floatX
        initializer = option["initializer"]
        regularizer = option["regularizer"]
        keep_prob = option["keep_prob"] or 1.0

        scope = option["scope"]
        encoder_scope = "encoder"
        block_encoder_scope = "block_encoder"
        decoder_scope = "decoder"

        encoder = Encoder(sedim, shdim)
        block_encoder = Block_Encoder(shdim * 2, bshdim)
        decoderType = eval("Decoder{}".format(option["decoder"]))
        decoder = decoderType(tedim, thdim, ahdim, 2 * shdim, bahdim, 2 * bshdim, dim_maxout=maxdim, max_part=maxpart, dim_readout=deephid,
                              n_y_vocab=tvsize)

        # training graph
        with ops.variable_scope(scope, initializer=initializer,
                                regularizer=regularizer, dtype=dtype):
            src_seq = T.imatrix("source_sequence")
            src_mask = T.matrix("source_sequence_mask")
            initid = T.matrix("source_init_id")
            index = T.imatrix("block_index")
            backindex = T.imatrix("back_block_index")
            imask = T.matrix("index_mask")
            tgt_seq = T.imatrix("target_sequence")
            tgt_mask = T.matrix("target_sequence_mask")
            mask = T.tensor3("mask")

            with ops.variable_scope("source_embedding"):
                source_embedding = ops.get_variable("embedding",
                                                    [svsize, sedim])
                source_bias = ops.get_variable("bias", [sedim])

            with ops.variable_scope("target_embedding"):
                target_embedding = ops.get_variable("embedding",
                                                    [tvsize, tedim])
                target_bias = ops.get_variable("bias", [tedim])

            source_inputs = nn.embedding_lookup(source_embedding, src_seq)
            target_inputs = nn.embedding_lookup(target_embedding, tgt_seq)

            source_inputs = source_inputs + source_bias
            target_inputs = target_inputs + target_bias

            if keep_prob < 1.0:
                source_inputs = nn.dropout(source_inputs, keep_prob=keep_prob)
                target_inputs = nn.dropout(target_inputs, keep_prob=keep_prob)

            states, r_states = encoder.forward(source_inputs, src_mask, initid, scope=encoder_scope)
            annotation = T.concatenate([states, r_states], 2)

            trans_foward_states = theano.tensor.transpose(states, (1, 0, 2))
            trans_foward_index = theano.tensor.transpose(index, (1, 0))
            sentence_foward_states = theano.tensor.transpose(trans_foward_states[theano.tensor.arange(trans_foward_index.shape[0])[:, None], trans_foward_index], (1, 0, 2))
            trans_backward_states = theano.tensor.transpose(r_states, (1, 0, 2))
            trans_backward_index = theano.tensor.transpose(backindex, (1, 0))
            sentence_backward_states = theano.tensor.transpose(trans_backward_states[theano.tensor.arange(trans_backward_index.shape[0])[:, None], trans_backward_index], (1, 0, 2))
            sentence_states = T.concatenate([sentence_foward_states,sentence_backward_states], 2)
            bstates, br_states = block_encoder.forward(sentence_states,imask, scope=block_encoder_scope)
            bannotation = T.concatenate([bstates, br_states],2)

            annotation = nn.dropout(annotation, keep_prob=keep_prob)
            #bannotation = nn.dropout(bannotation, keep_prob=keep_prob)

            # compute initial state for decoder
            # first state of backward encoder
            final_state = br_states[0]
            with ops.variable_scope(decoder_scope):
                initial_state = nn.feedforward(final_state, [bshdim, thdim],
                                               True, scope="block_initial",
                                               activation=T.tanh)
                # keys for query
                mapped_keys = map_key(annotation, 2 * shdim, ahdim, map_scope="word_level_map_key")
                mapped_block_keys = map_key(bannotation, 2 * bshdim, bahdim, map_scope="block_level_map_key")

                _, _, _, cost, _ = decoder.forward(tgt_seq, target_inputs, tgt_mask, mapped_keys, mask,
                                                    annotation, mapped_block_keys, imask, bannotation,
                                                     initial_state, keep_prob)


        training_inputs = [src_seq, src_mask, initid, index, backindex, imask, tgt_seq, tgt_mask, mask]
        training_outputs = [cost]

        # decoding graph
        with ops.variable_scope(scope, reuse=True):
            prev_words = T.ivector("prev_words")
            s2tmask = T.matrix("s2tmask")

            # disable dropout
            source_inputs = nn.embedding_lookup(source_embedding, src_seq)
            source_inputs = source_inputs + source_bias
            target_inputs = nn.embedding_lookup(target_embedding, tgt_seq)
            target_inputs = target_inputs + target_bias

            states, r_states = encoder.forward(source_inputs, src_mask, initid)

            trans_foward_states = theano.tensor.transpose(states, (1, 0, 2))
            trans_foward_index = theano.tensor.transpose(index, (1, 0))
            sentence_foward_states = theano.tensor.transpose(
                trans_foward_states[theano.tensor.arange(trans_foward_index.shape[0])[:, None], trans_foward_index],
                (1, 0, 2))
            trans_backward_states = theano.tensor.transpose(r_states, (1, 0, 2))
            trans_backward_index = theano.tensor.transpose(backindex, (1, 0))
            sentence_backward_states = theano.tensor.transpose(trans_backward_states[
                                                                   theano.tensor.arange(trans_backward_index.shape[0])[
                                                                   :, None], trans_backward_index], (1, 0, 2))
            sentence_states = T.concatenate([sentence_foward_states, sentence_backward_states], 2)

            bstates, br_states = block_encoder.forward(sentence_states, imask)

            annotation = T.concatenate([states, r_states], 2)
            bannotation = T.concatenate([bstates, br_states], 2)


            # decoder
            final_state = br_states[0]
            with ops.variable_scope(decoder_scope):
                initial_state = nn.feedforward(final_state, [bshdim, thdim],
                                               True, scope="block_initial",
                                               activation=T.tanh)

                mapped_keys = map_key(annotation, 2 * shdim, ahdim, map_scope='word_level_map_key')
                mapped_block_keys = map_key(bannotation, 2 * bshdim, bahdim, map_scope='block_level_map_key')
                #mapped_keys = map_key(annotation, 2 * shdim, ahdim)

            prev_inputs = nn.embedding_lookup(target_embedding, prev_words)
            prev_inputs = prev_inputs + target_bias

            cond = T.neq(prev_words, 0)
            # zeros out embedding if y is 0, which indicates <s>
            prev_inputs = prev_inputs * cond[:, None]

            with ops.variable_scope(decoder_scope):
                ymask = T.ones_like(prev_words, dtype=dtype)
                #y_prev, ymask, key_mask, state, keys, values, bkeys, bvalues, bkey_mask
                next_state, context, bcontext = decoder.step(prev_inputs, ymask, s2tmask, initial_state,
                                                   mapped_keys, annotation,
                                                   mapped_block_keys, imask, bannotation)

                if option["decoder"] == "GruSimple":
                    probs = decoder.prediction(prev_inputs, initial_state, context, bcontext)
                elif option["decoder"] == "GruCond":
                    probs = decoder.prediction(prev_inputs, next_state, context, bcontext)
        
        # encoding
        encoding_inputs = [src_seq, src_mask, initid, index, backindex, imask]
        encoding_outputs = [annotation, bannotation, initial_state, mapped_keys, mapped_block_keys]
        encode = theano.function(encoding_inputs, encoding_outputs)

        if option["decoder"] == "GruSimple":
            prediction_inputs = [prev_words, s2tmask, initial_state, annotation,
                                 mapped_keys, bannotation, mapped_block_keys, imask]
            prediction_outputs = [probs, context, bcontext]
            predict = theano.function(prediction_inputs, prediction_outputs)

            generation_inputs = [prev_words, initial_state, context, bcontext]
            generation_outputs = next_state
            generate = theano.function(generation_inputs, generation_outputs)

            self.predict = predict
            self.generate = generate
        elif option["decoder"] == "GruCond":
            prediction_inputs = [prev_words, s2tmask, initial_state, annotation,
                                 mapped_keys, bannotation, mapped_block_keys, imask]
            prediction_outputs = [probs, next_state]
            predict = theano.function(prediction_inputs, prediction_outputs)
            self.predict = predict
        '''
        # optional graph
        with ops.variable_scope(scope, reuse=True):
            sample = decoder.build_sampling(src_seq, src_mask, target_embedding, target_bias, mapped_keys,
                                            annotation, initial_state)
            align = decoder.build_attention(src_seq, src_mask, target_inputs, tgt_seq, tgt_mask, mapped_keys,
                                            annotation, initial_state)
            with ops.variable_scope(decoder_scope):
                initial_state = nn.feedforward(final_state, [shdim, thdim],
                                               True, scope="initial",
                                               activation=T.tanh)
                # keys for query
                mapped_keys = map_key(annotation, 2 * shdim, ahdim)

                _, _, _,snt_cost  = decoder.forward(tgt_seq, target_inputs, tgt_mask, mapped_keys, src_mask,
                                                    annotation, initial_state, 1.0)
            get_snt_cost = theano.function(training_inputs, snt_cost)
        '''

        self.cost = cost
        self.inputs = training_inputs
        self.outputs = training_outputs
        self.updates = []
        #self.align = align
        #self.sample = sample
        self.encode = encode

        #self.get_snt_cost = get_snt_cost
        self.option = option


# TODO: add batched decoding
def beamsearch(model, seq, initid, index, backindex, imask, s2tmask, beamsize=10, normalize=False,
               maxlen=None, minlen=None, arithmetic=False, dtype=None, suppress_unk=False):
    dtype = dtype or theano.config.floatX

    #if not isinstance(models, (list, tuple)):
        #models = [models]

    #num_models = len(models)

    # get vocabulary from the first model
    option = model.option
    vocab = option["vocabulary"][1][1]
    eosid = option["eosid"]
    bosid = option["bosid"]
    eobid = option["eobid"]
    #print eobid
    unk_sym = model.option["unk"]
    unk_id = option["vocabulary"][1][0][unk_sym]

    if maxlen is None:
        maxlen = seq.shape[0] * 3

    if minlen is None:
        minlen = seq.shape[0] / 2

    # encoding source
    #if xmask is None:
    xmask = numpy.ones(seq.shape, dtype)

    outputs = model.encode(seq, xmask, initid, index, backindex, imask)
    #annotation, bannotation, initial_state, mapped_keys, mapped_block_keys
    annotations = outputs[0]
    bannotations = outputs[1]
    states = outputs[2]
    mapped_annots = outputs[3]
    mapped_bannots = outputs[4]

    initial_beam = beam(beamsize)
    size = beamsize
    # bosid must be 0
    initial_beam.candidates = [[bosid]]
    initial_beam.scores = numpy.zeros([1], dtype)

    hypo_list = []
    beam_list = [initial_beam]
    done_predicate = lambda x: x[-1] == eosid
    done_block_predicate = lambda x: x[-1] == eobid

    mind = [0]
    sublen = index.shape[0]
    #print sublen
    flag = True

    for k in range(maxlen):
        # get previous results
        prev_beam = beam_list[-1]
        candidates = prev_beam.candidates
        num = len(candidates)
        last_words = numpy.array(map(lambda cand: cand[-1], candidates), "int32")

        # compute context first, then compute word distribution
        #batch_mask = numpy.repeat(xmask, num, 1)
        annots = numpy.repeat(annotations, num, 1)
        mannots = numpy.repeat(mapped_annots, num,1)
        bannots = numpy.repeat(bannotations, num, 1)
        mbannots = numpy.repeat(mapped_bannots, num, 1)
        imasks = numpy.repeat(imask, num, 1)
        m = numpy.array(s2tmask[mind]).reshape(num, s2tmask.shape[2])

        # prev_words, initial_state, annotation,mapped_keys, mask, bannotation, mapped_block_keys, imask
        # return probs, next_state
        outputs = model.predict(last_words, m, states, annots, mannots, bannots, mbannots, imasks)

        probs = outputs[0]

        # search nbest given word distribution
        if arithmetic:
            logprobs = numpy.log(probs)
        else:
            # geometric mean
            logprobs = numpy.log(probs)

        if suppress_unk:
            logprobs[:, unk_id] = -numpy.inf

        if k < minlen:
            logprobs[:, eosid] = -numpy.inf  # make sure eos won't be selected

        # force to add eos symbol
        if k == maxlen - 1:
            # copy
            eosprob = logprobs[:, eosid].copy()
            logprobs[:, :] = -numpy.inf
            logprobs[:, eosid] = eosprob  # make sure eos will be selected
            mind = [sublen for uu in mind]

        if (flag):
            mind = numpy.repeat(numpy.array(mind), size).tolist()
            flag = False
        next_beam = beam(size)
        finished, remain_beam_indices, mind = next_beam.prune(logprobs, done_predicate, done_block_predicate,
                                                        prev_beam, mind, sublen)

        hypo_list.extend(finished)  # completed translation
        size -= len(finished)

        if size == 0:  # reach k completed translation before maxlen
            break

        # generate next state
        candidates = next_beam.candidates
        num = len(candidates)
        last_words = numpy.array(map(lambda t: t[-1], candidates), "int32")

        if option["decoder"] == "GruSimple":
            contexts = [item[1] for item in outputs]
            states = select_nbest(states, remain_beam_indices)  # select corresponding states for each model
            contexts = select_nbest(contexts, remain_beam_indices)

            states = [model.generate(last_words, state, context)
                      for model, state, context in zip(models, states, contexts)]
        elif option["decoder"] == "GruCond":
            states = outputs[1]
            states = select_nbest(states, remain_beam_indices)  # select corresponding states for each model

        beam_list.append(next_beam)

    # postprocessing
    if len(hypo_list) == 0:
        score_list = [0.0]
        hypo_list = [[eosid]]
    else:
        score_list = [item[1] for item in hypo_list]
        # exclude bos symbol
        hypo_list = [item[0][1:] for item in hypo_list]

    for i, (trans, score) in enumerate(zip(hypo_list, score_list)):
        count = len(trans)
        if count > 0:
            if normalize:
                score_list[i] = score / count
            else:
                score_list[i] = score

    # sort
    hypo_list = numpy.array(hypo_list)[numpy.argsort(score_list)]
    score_list = numpy.array(sorted(score_list))

    output = []

    for trans, score in zip(hypo_list, score_list):
        trans = map(lambda x: vocab[x], trans)
        output.append((trans, score))

    return output


def batchsample(model, seq, mask, maxlen=None):
    sampler = model.sample

    vocabulary = model.option["vocabulary"]
    eosid = model.option["eosid"]
    vocab = vocabulary[1][1]

    if maxlen is None:
        maxlen = int(len(seq) * 1.5)

    words = sampler(seq, mask, maxlen)
    trans = words.astype("int32")

    samples = []

    for i in range(trans.shape[1]):
        example = trans[:, i]
        # remove eos symbol
        index = -1

        for i in range(len(example)):
            if example[i] == eosid:
                index = i
                break

        if index >= 0:
            example = example[:index]

        example = map(lambda x: vocab[x], example)

        samples.append(example)

    return samples


# used for analysis
def evaluate_model(model, xseq, xmask, yseq, ymask, alignment=None,
                   verbose=False):
    t = yseq.shape[0]
    batch = yseq.shape[1]

    vocab = model.option["vocabulary"][1][1]

    annotation, states, mapped_annot = model.encode(xseq, xmask)

    last_words = numpy.zeros([batch], "int32")
    costs = numpy.zeros([batch], "float32")
    indices = numpy.arange(batch, dtype="int32")

    for i in range(t):
        outputs = model.predict(last_words, states, annotation, mapped_annot,
                                xmask)
        # probs: batch * vocab
        # contexts: batch * hdim
        # alpha: batch * srclen
        probs, contexts, alpha = outputs

        if alignment is not None:
            # alignment tgt * src * batch
            contexts = numpy.sum(alignment[i][:, :, None] * annotation, 0)

        max_prob = probs.argmax(1)
        order = numpy.argsort(-probs)
        label = yseq[i]
        mask = ymask[i]

        if verbose:
            for i, (pred, gold, msk) in enumerate(zip(max_prob, label, mask)):
                if msk and pred != gold:
                    gold_order = None

                    for j in range(len(order[i])):
                        if order[i][j] == gold:
                            gold_order = j
                            break

                    ent = -numpy.sum(probs[i] * numpy.log(probs[i]))
                    pp = probs[i, pred]
                    gp = probs[i, gold]
                    pred = vocab[pred]
                    gold = vocab[gold]
                    print "%d: predication error, %s vs %s" % (i, pred, gold)
                    print "prob: %f vs %f, entropy: %f" % (pp, gp, ent)
                    print "gold is %d-th best" % (gold_order + 1)

        costs -= numpy.log(probs[indices, label]) * mask

        last_words = label
        states = model.generate(last_words, states, contexts)

    return costs
