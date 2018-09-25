# rnn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano

from dropout import dropout
from nn import linear, gate_linear, feedforward, gate_feedforward, ln_linear
from ops import variable_scope


class gate_rnn_cell(object):
    def __call__(self, context, bcontext, state, scope=None):
        raise NotImplementedError("abstract method")

    def zero_state(self, batch_size, dtype):
        raise NotImplementedError("abstract method")


class gate_gru_cell(gate_rnn_cell):
    def __init__(self, context_size, bcontext_size, output_size):
        self.output_size = output_size
        self.context_size = context_size
        self.bcontext_size = bcontext_size

    def __call__(self, context, bcontext, state, scope=None):
        output_size = self.output_size
        context_size = self.context_size
        bcontext_size = self.bcontext_size

        size = [[output_size] + [context_size] + [bcontext_size], output_size]

        with variable_scope(scope or "gate_gru_cell"):
            new_inputs = [state]
            g = feedforward(new_inputs + [context] + [bcontext],size,False, scope="context_gate")
            r = gate_feedforward(new_inputs, context, bcontext, g,
                                 [output_size], context_size, bcontext_size, output_size,
                                 False, scope="reset_gate")
            u = gate_feedforward(new_inputs, context, bcontext, g,
                                 [output_size], context_size, bcontext_size, output_size,
                                 False, scope="update_gate")
            new_inputs = [r * state]
            c = gate_feedforward(new_inputs, context, bcontext, g,
                                 [output_size], context_size, bcontext_size, output_size,
                                 True, activation=theano.tensor.tanh, scope="candidate")

            new_state = (1.0 - u) * state + u * c

        return new_state, new_state

    def zero_state(self, batch_size, dtype=None):
        output_size = self.output_size
        return theano.tensor.zeros([batch_size, output_size], dtype=dtype)





