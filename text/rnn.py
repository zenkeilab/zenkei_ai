# from latest fastai/text/models.py

import warnings

import torch
from torch import nn
import torch.nn.functional as F


def range_of(x): return torch.arange(len(x))


# bn_drop_lin() is from the latest fastai/layers.py
def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None):
    "`n_in`->bn->dropout->linear(`n_in`,`n_out`)->`actn`"
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers




def dropout_mask(x, sz, p):
    "Return a dropout mask of the same type as x, size sz, with probability p to cancel an element."
    return x.new(*sz).bernoulli_(1-p).div_(1-p)

class RNNDropout(nn.Module):
    "Dropout that is consistent on the seq_len dimension."

    def __init__(self, p=0.5):
        super().__init__()
        self.p=p

    def forward(self, x):
        if not self.training or self.p == 0.: return x
        m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
        return x * m


class WeightDropout(nn.Module):
    "A module that warps another layer in which some weights will be replaced by 0 during training."

    def __init__(
        self, module, weight_p, layer_names=['weight_hh_l0']):

        super().__init__()

        self.module = module
        self.weight_p = weight_p
        self.layer_names = layer_names

        for layer in self.layer_names:
            #Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()


class EmbeddingDropout(nn.Module):
    "Apply dropout in the embedding layer by zeroing out some elements of the embedding vector."

    def __init__(self, emb, embed_p):
        super().__init__()
        self.emb,self.embed_p = emb,embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None: self.pad_idx = -1

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0),1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else: masked_embed = self.emb.weight
        if scale: masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


class RNNCore(nn.Module):
    "AWD-LSTM by https://arxiv.org/abs/1708.02182."

    initrange=0.1

    def __init__(
        self,
        bs, vs, em_sz, nh, nl, pad_token,
        bidir=False,
        hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5):

        super().__init__()

        self.vs = vs
        self.em_sz = em_sz
        self.nh = nh
        self.nl = nl
        self.bs = bs

        self.ndir = 2 if bidir else 1

        self.encoder = nn.Embedding(vs, em_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)

        self.rnns = [
            nn.LSTM(
                em_sz if l == 0 else nh,
                (nh if l != nl - 1 else em_sz)//self.ndir,
                1,
                dropout=0,
                bidirectional=bidir)
            for l in range(nl)
        ]
        if weight_p != 0:
            self.rnns = [
                WeightDropout(rnn, weight_p)
                for rnn in self.rnns
            ]
        self.rnns = nn.ModuleList(self.rnns)

        self.hidden_dps = nn.ModuleList(
            [RNNDropout(hidden_p)
             for l in range(nl)]
        )

        #self.decoder = nn.Linear(em_sz, vs)

        self.init_hidden()

    def forward(self, input):
        # input.shape is [sl, bs]
        sl, bs = input.size()
        if self.bs != bs:
            self.bs = bs
            self.init_hidden()

        raw_outputs, hiddens, outputs = [], [], []
        outp = self.input_dp(self.encoder_dp(input))
        for l, (rnn, h_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            outp, hidden = rnn(outp, self.h[l])
            raw_outputs.append(outp)
            hiddens.append(hidden)

            if l != self.nl - 1:
                outp = h_dp(outp)
            outputs.append(outp)


        # drop the dependencies
        self.h = [
            (h[0].detach(), h[1].detach())
            for h in hiddens
        ]

        #dec = self.decoder(outputs[-1])
        #return dec.view(-1, self.vs)
        return raw_outputs, outputs


    def init_hidden(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.h = [
            (torch.zeros(
                self.ndir, self.bs,
                (self.nh if l != self.nl - 1 else self.em_sz)//self.ndir,
                requires_grad=True, device=device),
             torch.zeros(
                 self.ndir, self.bs,
                 (self.nh if l != self.nl - 1 else self.em_sz)//self.ndir,
                 requires_grad=True, device=device)
            )
            for l in range(self.nl)
        ]

    def reset(self):
        self.init_hidden()

class LinearDecoder(nn.Module):
    "To go on top of a RNNCore module and create a Language Model."

    initrange=0.1

    def __init__(self, n_out, n_hid, output_p, tie_encoder=None, bias=True):
        super().__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.output_dp = RNNDropout(output_p)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.output_dp(outputs[-1])
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded, raw_outputs, outputs

class SequentialRNN(nn.Sequential):
    "A sequential module that passes the reset call to its children."
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

class MultiBatchRNNCore(RNNCore):
    "Create a RNNCore module that can process a full sentence."

    def __init__(self, bptt, max_seq, *args, **kwargs):
        self.bptt = bptt
        self.max_seq = max_seq
        super().__init__(*args, **kwargs)

    def concat(self, arrs):
        "Concatenate the `arrs` along the batch dimension."
        return [torch.cat([l[si] for l in arrs]) for si in range_of(arrs[0])]

    def forward(self, input):
        sl, bs = input.size()
        self.reset()
        raw_outputs, outputs = [], []
        for i in range(0, sl, self.bptt):
            r, o = super().forward(input[i: min(i+self.bptt, sl)])
            if i>(sl-self.max_seq):
                raw_outputs.append(r)
                outputs.append(o)
        return self.concat(raw_outputs), self.concat(outputs)

class PoolingLinearClassifier(nn.Module):
    "Create a linear classifier with pooling."

    def __init__(self, layers, drops):
        super().__init__()
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in,n_out,p,actn in zip(layers[:-1],layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)

    def pool(self, x, bs, is_max):
        "Pool the tensor along the seq_len dimension."
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)

    def forward(self, input):
        raw_outputs, outputs = input
        output = outputs[-1]
        sl,bs,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        x = self.layers(x)
        return x, raw_outputs, outputs


def get_language_model(
    bs, vocab_sz, emb_sz, n_hid, n_layers, pad_token,
    tie_weights=True, bias=True, bidir=False,
    output_p=0.4, hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5):
    "Create a full AWD-LSTM."
    rnn_enc = RNNCore(
        bs, vocab_sz, emb_sz, n_hid, n_layers, pad_token,
        bidir=bidir,
        hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    enc = rnn_enc.encoder if tie_weights else None
    return SequentialRNN(
        rnn_enc,
        LinearDecoder(vocab_sz, emb_sz, output_p, tie_encoder=enc, bias=bias))

def get_rnn_classifier(
    bs, bptt, max_seq, n_class, vocab_sz, emb_sz, n_hid, n_layers, pad_token,
    layers, drops, bidir=False,
    hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5):
    "Create a RNN classifier model."
    rnn_enc = MultiBatchRNNCore(
        bptt, max_seq,
        bs, vocab_sz, emb_sz, n_hid, n_layers, pad_token,
        bidir=bidir,
        hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    return SequentialRNN(
        rnn_enc,
        PoolingLinearClassifier(layers, drops))
