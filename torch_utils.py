import copy
import math
import gc
from tqdm import tqdm
from tqdm import tqdm_notebook
from itertools import chain

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os.path
import joblib

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from functools import partial

import matplotlib.animation as animation
import torchvision.utils as vutils




# one-cycle scheduling related routines
def interp_lin(dx, y0, y1):
    return y0 + dx * (y1 - y0)

def interp_cos(dx, y0, y1):
    return y0 + 0.5 * (1.0-math.cos(dx * math.pi)) * (y1 - y0)


class OneCycleGenerator():
    '''
    xs : list
        List of keypoints for the indepenent variable, that is,
        ratio of the current time from the starting time
        to the final time, so that it is in the range `[0, 1]`.
    ys : list
        List of keypoints for the depentne variable, like
        learning rate `lr` or momentum `mom` at each keypoint
        in `xs`.
    interp_fun : function
        Interporation function getting one argument `pos` in the range `[0, 1]`.
    '''
    def __init__(
        self,
        xs, ys,
        interp_fun=interp_cos
    ):
        if len(xs) != len(ys):
            raise ValueError('len(xs) and len(ys) must be the same')
        if len(xs) < 2:
            raise ValueError('len(xs) must be equal or greater than 2')

        self.xs = xs
        self.ys = ys
        self.interp_fun = interp_fun

    def __call__(self, pos):
        return self.interp(pos)

    def interp(self, pos):
        if pos < 0.0:   pos = 0.0; i_phase = 0
        elif pos > 1.0: pos = 1.0; i_phase = len(self.xs)-1
        else:
            for i_phase, x in enumerate(self.xs[1:]):
                if pos < x: break

        t = (pos - self.xs[i_phase]) / (self.xs[i_phase+1]-self.xs[i_phase])
        return self.interp_fun(t, self.ys[i_phase], self.ys[i_phase+1])




# label smoothing loss function
# from fast.ai Lesson 12 https://youtu.be/vnOpEwmtFJ8?t=1292
def _lin_comb(v1, v2, beta): return beta*v1 + (1-beta)*v2

def _reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = _reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return _lin_comb(loss/c, nll, self.epsilon)




def accuracy(outputs, labels):
    _, preds = torch.max(outputs, -1)
    acc = torch.mean((preds == labels.data).float())
    return acc.item()


def seq2seq_reg(output, xtra, loss, alpha=0, beta=0):
    hs,dropped_hs = xtra
    if alpha:  # Activation Regularization
        loss = loss + (alpha * dropped_hs[-1].pow(2).mean()).sum()
    if beta:   # Temporal Activation Regularization (slowness)
        h = hs[-1]
        if len(h)>1: loss = loss + (beta * (h[1:] - h[:-1]).pow(2).mean()).sum()
    return loss


# from fastai library

# for freeze_to() and unfreeze()
def _children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())

def _set_trainable_attr(m, b, wo_bn=True):
    if wo_bn:
        # always trainable for batch norm layers
        if isinstance(m, (
            nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d
        )): b = True

    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b

def _apply_leaf(m, f):
    c = _children(m)
    if isinstance(m, nn.Module): f(m)
    if len(c) > 0:
        for l in c: _apply_leaf(l,f)

def _set_trainable(l, b, wo_bn=True):
    _apply_leaf(l, lambda m: _set_trainable_attr(m, b, wo_bn=wo_bn))


# for discriminative learning rate training
def _trainable_params(m):
    '''Returns a list of trainable parameters in the model m.
    (i.e., those that require gradients.)'''
    return [p for p in m.parameters() if p.requires_grad]

def _chain_params(p):
    if isinstance(p, (list,tuple)):
        return list(chain(*[_trainable_params(o) for o in p]))
    return _trainable_params(p)


# for mixup
# from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
def mixup_data(x, y, device, alpha=1.0, max_lam=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        # modification based on fastai
        if max_lam: lam = max(lam, 1 - lam)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# static methods for model class are just listed here

def plot_lrf(
    lrf_data, ylim=None, xlim=None, ax=None,
    with_points=True, with_errorbars=True):
    """Plot the reslut of the learning rate finder lrf_data.

    Parameters
    ----------
    lrf_data : list/tuple or str
        The result of LR finder, either python dict obtained by `get_lrf_data`
        or file path dumped by `joblib.dump`.
    ylim: List of Floats (optional)
        The range of the plot in y axis.
    xlim: List of Floats (optional)
        The range of the plot in x axis.
    with_points : Bool
        Plot points (for the result obtained with `ave_steps > 1`).
        Default is `True`.
    with_errorbars : Bool
        Plot errorbars (for the result obtained with `ave_steps > 1`).
        Default is `True`.
    """

    if isinstance(lrf_data, str):
        lrf_data = joblib.load(lrf_data)

    if len(lrf_data['lrs']) <= 0 or len(lrf_data['losses']) <= 0: return

    if not ax: fig, ax = plt.subplots(figsize=(14, 10))

    ax.semilogx(lrf_data['lrs'], lrf_data['losses'])
    try:
        if (isinstance(ylim, tuple) or isinstance(ylim, list)) and len(ylim) == 2:
            ax.set_ylim(ylim[0], ylim[1])
        if xlim == 'min':
            ax.set_xlim(lrf_data['lrf_x_range_min'][0], lrf_data['lrf_x_range_min'][1])
            if ylim is None:
                ylim_min = min(lrf_data['losses'][:lrf_data['lrf_x_range_min']])
                ylim_max = max(lrf_data['losses'][:lrf_data['lrf_x_range_min']])
                ax.set_ylim(ylim_min, ylim_max)
        elif xlim == 'max':
            ax.set_xlim(lrf_data['lrf_x_range_max'][0], lrf_data['lrf_x_range_max'][1])
            if ylim is None:
                ylim_min = min(lrf_data['losses'][:lrf_data['lrf_x_range_max']])
                ylim_max = max(lrf_data['losses'][:lrf_data['lrf_x_range_max']])
                ax.set_ylim(ylim_min, ylim_max)
        else:
            ax.set_xlim(xlim[0], xlim[1])
    except:
        pass

    if lrf_data['lrf_ave_steps'] > 1:
        if with_points:
            for i in range(lrf_data['lrf_ave_steps']):
                ax.semilogx(lrf_data['lrf_ave_samples'][i][0],
                            lrf_data['lrf_ave_samples'][i][1], 'o')
        if with_errorbars:
            #ax = ax.gca() # get current axes
            ax.set_xscale('log')
            ax.errorbar(lrf_data['lrs'], lrf_data['losses'], yerr=lrf_data['losses_std'])

    ax.grid(which='major',color='k',linestyle='--')

    return ax


def print_model_size(model):
    params = list(model.parameters())
    num_params_all = sum([np.prod(np.array(params[i].size())) for i in range(len(params))])

    num_params_train = 0
    num_params_nontrain = 0
    for param in model.parameters():
        n = np.prod(np.array(param.size()))
        if param.requires_grad:
            num_params_train += n
        else:
            num_params_nontrain += n

    print('Total params: {:,}'.format(num_params_all))
    print('Trainable params: {:,}'.format(num_params_train))
    print('Non-trainable params: {:,}'.format(num_params_nontrain))


def plot_history(history):
    n_cols = 2
    n_rows = 1

    plt.figure(figsize=(10 * n_cols, 8 * n_rows))

    if 'loss_all' in history:
        n_all = len(history['loss_all'])
        n_epochs = len(history['loss'])
        n_steps = n_all // len(history['loss'])
        x = (np.array(range(n_epochs)) + 0.5)*n_steps

    # loss graph
    plt.subplot(n_rows, n_cols, 1)
    plt.title('loss')
    plt.ylabel('loss')
    plt.grid(which='major',color='k',linestyle='--')
    if 'loss_all' in history:
        plt.xlabel('step')
        plt.plot(history['loss_all'], '.', zorder=1)
        if len(history['loss']) > 1:
            plt.scatter(x, history['loss'], s=400, c='orange', zorder=2)
            plt.plot(x, history['loss'], '-',
                     label='training')
        else:
            plt.scatter(x, history['loss'], s=400, c='orange', zorder=2,
                        label='training')

        if 'val_loss' in history:
            if len(history['loss']) > 1:
                plt.scatter(x, history['val_loss'], s=400, c='red', zorder=2)
                plt.plot(x, history['val_loss'], '-', c='red',
                         label='validation')
            else:
                plt.scatter(x, history['val_loss'], s=400, c='red', zorder=2,
                            label='validation')
    else:
        plt.xlabel('epoch')
        if len(history['loss']) > 1:
            plt.plot(history['loss'], 'o', c='orange')
            plt.plot(history['loss'], '-', c='orange',
                     label='training')
        else:
            plt.plot(history['loss'], 'o', c='orange',
                     label='training')
        if 'val_loss' in history:
            if len(history['loss']) > 1:
                plt.plot(history['val_loss'], 'o', c='red')
                plt.plot(history['val_loss'], '-', c='red',
                         label='validation')
            else:
                plt.plot(history['val_loss'], 'o', c='red',
                         label='validation')
    plt.legend()


    # accuracy graph
    if 'acc' in history:
        plt.subplot(n_rows, n_cols, 2)
        plt.title('acc')
        plt.ylabel('acc')
        plt.grid(which='major',color='k',linestyle='--')
        if 'acc_all' in history:
            plt.xlabel('step')
            plt.plot(history['acc_all'], '.', zorder=1)
            if len(history['acc']) > 1:
                plt.scatter(x, history['acc'], s=400, c='orange', zorder=2)
                plt.plot(x, history['acc'], '-',
                         label='training')
            else:
                plt.scatter(x, history['acc'], s=400, c='orange', zorder=2,
                            label='training')

            if 'val_acc' in history:
                if len(history['acc']) > 1:
                    plt.scatter(x, history['val_acc'], s=400, c='red', zorder=2)
                    plt.plot(x, history['val_acc'], '-', c='red',
                             label='validation')
                else:
                    plt.scatter(x, history['val_acc'], s=400, c='red', zorder=2,
                                label='validation')
        else:
            plt.xlabel('epoch')
            if len(history['acc']) > 1:
                plt.plot(history['acc'], 'o', c='orange')
                plt.plot(history['acc'], '-', c='orange',
                         label='training')
            else:
                plt.plot(history['acc'], 'o', c='orange',
                         label='training')

            if 'val_acc' in history:
                if len(history['acc']) > 1:
                    plt.plot(history['val_acc'], 'o', c='red')
                    plt.plot(history['val_acc'], '-', c='red',
                             label='validation')
                else:
                    plt.plot(history['val_acc'], 'o', c='red',
                             label='validation')
        plt.legend()

    elif 'val_acc' in history:
        plt.subplot(n_rows, n_cols, 2)
        plt.title('acc')
        plt.ylabel('acc')
        plt.grid(which='major',color='k',linestyle='--')
        if 'loss_all' in history:
            # that is, x is defined
            plt.xlabel('step')
            if len(history['val_acc']) > 1:
                plt.scatter(x, history['val_acc'], s=400, c='red', zorder=2)
                plt.plot(x, history['val_acc'], '-', c='red',
                         label='validation')
            else:
                plt.scatter(x, history['val_acc'], s=400, c='red', zorder=2,
                            label='validation')
        else:
            plt.xlabel('epoch')
            if len(history['val_acc']) > 1:
                plt.plot(history['val_acc'], 'o', c='red')
                plt.plot(history['val_acc'], '-', c='red',
                         label='validation')
            else:
                plt.plot(history['val_acc'], 'o', c='red',
                         label='validation')
        plt.legend()

    plt.show()


def average_in_window(x, n):
    cnt = [0]*len(x)
    z = [0.0]*len(x)
    for i, y in enumerate(x):
        for j in range(n):
            k = i + j
            if k >= len(x): break
            z[i+j] += x[i]
            cnt[i+j] += 1
    for i in range(len(z)):
        if cnt[i] > 0: z[i] /= float(cnt[i])
    return z


def plot_histories(
        hs, labels=None, scale_fixed=False, xrange=None,
        n_ave=0, logscale=False):

    if len(hs) <= 0: return
    keys = hs[0].keys()
    if len(keys) <= 0: return

    for h in hs:
        if keys != h.keys():
            print('entries of hs are mismatched')
            return

    if n_ave > 1:
        aves = {k: [] for k in keys}
        for k, a in zip(keys, aves):
            for h in hs:
                a.append(model.average_in_window(h[k], n_ave))

    if logscale:
        plot_func = plt.semilogy
    else:
        plot_func = plt.plot

    n_cols = 2
    n_rows = len(keys) // 2 if len(keys)%2==0 else len(keys)+1

    plt.figure(figsize=(10 * n_cols, 8 * n_rows))

    for j, k in enumerate(keys):
        plt.subplot(n_rows, n_cols, j+1)
        for i, h in enumerate(hs):
            if n_ave > 1:
                if xrange is None: plot_func(aves[k][i])
                else: plot_func(aves[k][i][xrange[0]:xrange[1]])
            else:
                if xrange is None: plot_func(h[k])
                else: plot_func(h[k][xrange[0]:xrange[1]])

        plt.title(k)
        plt.ylabel(k)
        plt.xlabel('epoch')
        if not labels is None: plt.legend(labels, loc='lower right')
        plt.grid(which='major',color='k',linestyle='--')

    plt.show()


class model():
    """ZENKEI model class

    Parameters
    ----------
    model : torch.nn.Module
        The model to train, predict, etc.
    device: torch.device
        Set torch.device('cuda') or torch.device('cpu')
    opt_func: torch.optim
        The optimizer class for training. Note that this is NOT an instance.
    loss_dict: dictionary of the label (key) and the loss or metric function.
        For example, {'loss': F.cross_entropy, 'acc': accuracy}.
        The entry 'loss' is mandatory.
    layer_groups : None or list/tuple
        Layer groups for which different LRs and/or momentums can be set.
    is_rnn : bool
        Set `True` for the RNN model which outputs extra elements.
    reg_seq2seq : bool
        This is only for RNN models.
        If `True`, use regularization `reg_seq2seq`.
        Default is `False`.
    """

    def __init__(
        self,
        model,
        device,
        opt_func,
        loss_dict,
        layer_groups=None,
        is_rnn=False,
        reg_seq2seq=False):
        #reg_func=None):

        self.model = model
        self.device = device
        self.opt_func = opt_func
        self.loss_dict = loss_dict
        if not 'loss' in self.loss_dict:
            self.loss_dict['loss'] = F.cross_entropy
        if layer_groups is None:
            self.layer_groups = [self.model]
        else:
            self.layer_groups = layer_groups
        self.is_rnn = is_rnn
        #self.reg_func = reg_func
        self.reg_func = partial(seq2seq_reg, alpha=2, beta=1) if reg_seq2seq else None


        # for LR finder
        self.lrf_ave_steps = 1
        self.lrf_ave_samples = []
        self.lrs = None
        self.losses = None
        self.losses_std = None
        self.lrf_x_range_min = None
        self.lrf_x_range_max = None

        # for fit
        self.best_weights = None
        self.best_loss = 0.0


    def freeze_to(self, n, wo_bn=True):
        for l in self.layer_groups[:n]:
            _set_trainable(l, False, wo_bn=wo_bn)
        for l in self.layer_groups[n:]:
            _set_trainable(l, True)

    def unfreeze(self):
        self.freeze_to(0)


    # for weight decay in L2 regularization
    def _opt_params(self, lrs, wds):
        assert(len(self.layer_groups) == len(lrs))
        assert(len(self.layer_groups) == len(wds))

        params = list(zip(self.layer_groups, lrs, wds))
        return [
            {'params': _chain_params(parm),
             'lr': lr,
             'weight_decay': wd
            }
            for parm, lr, wd in params
        ]

    def _init_optimizer(self, lrs, wds):
        opt_params = self._opt_params(lrs, wds)
        return self.opt_func(opt_params)


    # for weight decay in AdamW type implementation
    def _opt_params_(self, lrs):
        assert(len(self.layer_groups) == len(lrs))

        params = list(zip(self.layer_groups, lrs))
        return [
            {'params': _chain_params(parm),
             'lr': lr
            }
            for parm, lr in params
        ]

    def _init_optimizer_(self, lrs):
        opt_params = self._opt_params_(lrs)
        return self.opt_func(opt_params)


    def _normalize_hyperparameters(self, lrs, wds):
        # for lrs
        if isinstance(lrs, (list, tuple)):
            if len(lrs) != len(self.layer_groups):
                raise ValueError('lrs - length is wrong')
            lrs0 = lrs.copy()
        elif isinstance(lrs, np.ndarray):
            if len(lrs.shape) != 1 or lrs.shape[0] != len(self.layer_groups):
                raise ValueError('lrs - length is wrong')
            lrs0 = lrs.copy()
        elif isinstance(lrs, (int, float)):
            lrs0 = [lrs] * len(self.layer_groups)
        else:
            raise ValueError('lrs must be either value or list of values')

        # for wds
        if wds is None:
            wds0 = [0.0] * len(self.layer_groups)
        elif isinstance(wds, (list, tuple)):
            if len(wds) != len(self.layer_groups):
                raise ValueError('wds - length is wrong')
            wds0 = wds.copy()
        elif isinstance(wds, np.ndarray):
            if len(wds.shape) != 1 or wds.shape[0] != len(self.layer_groups):
                raise ValueError('wds - length is wrong')
            wds0 = wds.copy()
        elif isinstance(wds, (int, float)):
            wds0 = [wds] * len(self.layer_groups)
        else:
            raise ValueError('wds must be either None, value or list of values')

        return lrs0, wds0




    def _lrf_finalize(self):
        # averaging process
        n_samples = [len(lr) for lr, loss in self.lrf_ave_samples]
        n_samples_max = max(n_samples)
        n_samples_min = min(n_samples)

        self.losses = np.empty((0))
        self.lrs = np.empty((0))
        for i in range(n_samples_max):
            n = 0
            ave_loss = 0.0
            lr_ = 0.0
            for lr, loss in self.lrf_ave_samples:
                try:
                    ave_loss += loss[i]
                    lr_ = lr[i]
                    n += 1
                except:
                    pass
            self.losses = np.concatenate((self.losses, [ave_loss / float(n)]))
            self.lrs = np.concatenate((self.lrs, [lr_]))

        self.losses_std = np.empty((0))
        for i in range(n_samples_max):
            n = 0
            var_loss = 0.0
            for lr, loss in self.lrf_ave_samples:
                try:
                    d = loss[i] - self.losses[i]
                    var_loss += d*d
                    n += 1
                except:
                    pass
            self.losses_std = np.concatenate(
                (self.losses_std, [math.sqrt(var_loss / float(n))]))

        self.lrf_x_range_min = (self.lrs[0], self.lrs[n_samples_min-1])
        self.lrf_x_range_max = (self.lrs[0], self.lrs[n_samples_max-1])


    def find_lr(
        self,
        trn_iter,
        ave_steps=1,
        lr_steps=10,
        lr_factor=1.1,
        max_lr=None,
        lrs=1.0e-6,
        wds=1e-7,
        wd_type=0,
        clip=1,
        grad_acc=1,
        mixup=False,
        mixup_alpha=1.0,
        mixup_max_lam=False,
        verbose=0):
        """Find learning rate parameter.

        Parameters
        ----------
        trn_iter:
            Data loader (iterator) providing input data as well as the labels.
        ave_steps: Int
            Number of runs to take ensemble average of the results.
            If 0 is given, losses are evalulated in a continuous mode,
            where the weights are not reset for each `lr`.
            Default is 1, that is, no ensemble averaging.
        lr_steps: Int (optional)
            Steps to optimize for each lr value. The default value is 10.
        lr_factor: Float (optional)
            The multiplicative factor for the learning rate
            for the next cycle. The default value is 1.1.
        max_lr: Float (optional)
            The maximum lr value to search.
            If None is given, searching is terminated by the loss limit
            which is 4 times bigger than the initial loss value.
            Default is None.
        lrs : Float or list/tuple/np.array of Floats (optional)
            The initial value of the learning rates for LR finder.
            If list/tuple/np.array is given, the length must be equal to
            the number of layer groups of the model and each value
            corresponds to the layer group.
            Default is 1.0e-6.
        wds : Float or list/tuple/np.array of Floats (optional)
            Weight decay factor.
            If list/tuple/np.array is given, the length must be equal to
            the number of layer groups of the model and each value
            corresponds to the layer group.
            The default value is 1e-7.
        wd_type : Int (optional)
            If `0` is given, AdamW type weight decay is used.
            Otherwise, conventional L2 regularization type is used.
        clip: None or Float (optional)
            Gradient clipping value.
            If None is given, gradient clipping is not applied.
            Default is None.
        grad_acc : Int
            Gradient accumulation factor.
            For example, if `2` is given, updating parameters are applied
            once in 2 steps, so that effective batch size becomes twice.
            Default is 1.
        mixup : bool
            Set `True` to use mixup data augmentation.
        mixup_alpha : Float (optional)
            Mixup parameter `alpha`. Only effective if `mixup=True`.
        mixup_max_lam : bool
            Set `True` to apply `max(lam, 1-lam)` for the parameter `lam`.
        verbose: Int
            Set non zero value for verbose mode.
        """

        # for house keeping
        init_state_dict = copy.deepcopy(self.model.state_dict())

        continuous_mode = False
        if ave_steps == 0:
            continuous_mode = True
            ave_steps = 1

        self.lrf_ave_steps = ave_steps
        self.lrf_ave_samples = []

        average_count = 0
        _lrs = []
        _losses = []
        lr_ratio = 1 / lr_factor # to begin with lr_ratio = 1
        init_loss = None

        gen = iter(trn_iter)
        loss_func = self.loss_dict['loss']

        lrs0, wds0 = self._normalize_hyperparameters(lrs, wds)
        if wd_type == 0:
            # AdamW
            optimizer = self._init_optimizer_(lrs0)
        else:
            # L2 regularization
            optimizer = self._init_optimizer(lrs0, wds0)


        if max_lr is None: lr_range = None
        else: lr_range = max_lr / lrs0[-1]


        # training loop
        self.model.train(mode=True)

        while 1:
            lr_ratio *= lr_factor

            for lr0, pg in zip(lrs0, optimizer.param_groups):
                pg['lr'] = lr0 * lr_ratio

            if hasattr(self.model, 'reset'):
                self.model.reset()

            # training loop with the learning rate
            if not continuous_mode:
                self.model.load_state_dict(init_state_dict)
            for loop in range(lr_steps):

                # zero the parameter gradients
                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                loss_acc = 0

                # gradient accumulation loop
                for ga_i in range(grad_acc):
                    try:
                        inputs, labels = next(gen)
                    except StopIteration:
                        gen = iter(trn_iter)
                        inputs, labels = next(gen)

                    #inputs = inputs.to(self.device)
                    #labels = labels.to(self.device)
                    if isinstance(inputs, (list, tuple)):
                        inputs = [i.to(self.device) for i in inputs]
                    else:
                        inputs = inputs.to(self.device)
                    if isinstance(labels, (list, tuple)):
                        labels = [l.to(self.device) for l in labels]
                    else:
                        labels = labels.to(self.device)

                    if mixup:
                        inputs, labels_a, labels_b, lam = mixup_data(
                            inputs, labels, self.device,
                            mixup_alpha, mixup_max_lam)

                    # forward
                    with torch.set_grad_enabled(True):
                        outputs = self.model(inputs)
                        if self.is_rnn and isinstance(outputs, tuple):
                            outputs, *xtra = outputs

                        if mixup:
                            loss = mixup_criterion(
                                loss_func, outputs,
                                labels_a, labels_b, lam)
                        else:
                            loss = loss_func(outputs, labels)
                        raw_loss_ = loss.item()

                        if self.is_rnn and not self.reg_func is None:
                            loss = self.reg_func(outputs, xtra, loss)

                        loss = loss / grad_acc

                        # backward -- acculumating the grads
                        loss.backward()

                    loss_acc += raw_loss_ / grad_acc

                # update parameters
                with torch.set_grad_enabled(True):
                    # Gradient clipping
                    if clip:
                        nn.utils.clip_grad_norm_(
                            _trainable_params(self.model), clip)

                    # weight decay
                    if wd_type == 0:
                        # AdamW
                        for wd, pg in zip(wds0, optimizer.param_groups):
                            for p in pg['params']:
                                p.data = p.data.add(-wd * pg['lr'], p.data)

                    # zero the parameter gradients
                    optimizer.step()


            # the last loss (after lr_steps)
            cur_loss = loss_acc
            if init_loss is None: init_loss = cur_loss

            # representative lr
            lr = lr_ratio * lrs0[-1]

            _lrs.append(lr)
            _losses.append(cur_loss)

            if verbose != 0:
                print('\rave_count: %d, lr: %e, loss: %f'\
                    % (average_count, lr, cur_loss), end='')

            if math.isinf(cur_loss) or math.isnan(cur_loss) or\
               (lr_range is None and cur_loss > 4.0 * init_loss) or\
               (not lr_range is None and lr_ratio > lr_range):

                self.lrf_ave_samples.append((_lrs, _losses))

                # house keeping
                self.model.load_state_dict(init_state_dict)

                average_count += 1
                if average_count >= ave_steps:
                    # averaging process
                    self._lrf_finalize()
                    return

                # for restart
                _lrs = []
                _losses = []
                lr_ratio = 1.0
                init_loss = None


    def plot_lr(self, ylim=None,
                figsize=(14, 10), with_points=False, with_errorbars=False):
        """Plot the reslut of the learning rate finder 'find_lr()'.

        Parameters
        ----------
        ylim: List of Floats (optional)
            The range of the plot in y axis.
        """

        if len(self.lrs) <= 0 or len(self.losses) <= 0: return

        if not figsize is None:
            plt.figure(figsize=figsize)

        plt.semilogx(self.lrs, self.losses)
        try:
            if (isinstance(ylim, tuple) or isinstance(ylim, list)) and len(ylim) == 2:
                plt.ylim(ylim[0], ylim[1])
            if xlim == 'min':
                plt.xlim(self.lrf_x_range_min[0], self.lrf_x_range_min[1])
                if ylim is None:
                    ylim_min = min(self.losses[:self.lrf_x_range_min])
                    ylim_max = max(self.losses[:self.lrf_x_range_min])
                    plt.ylim(ylim_min, ylim_max)
            elif xlim == 'max':
                plt.xlim(self.lrf_x_range_max[0], self.lrf_x_range_max[1])
                if ylim is None:
                    ylim_min = min(self.losses[:self.lrf_x_range_max])
                    ylim_max = max(self.losses[:self.lrf_x_range_max])
                    plt.ylim(ylim_min, ylim_max)
            else:
                plt.xlim(xlim[0], xlim[1])
        except:
            pass

        if self.lrf_ave_steps > 1:
            if with_points:
                for i in range(self.lrf_ave_steps):
                    plt.semilogx(self.lrf_ave_samples[i][0],
                                 self.lrf_ave_samples[i][1], 'o')
            if with_errorbars:
                ax = plt.gca() # get current axes
                ax.set_xscale('log')
                plt.errorbar(self.lrs, self.losses, yerr=self.losses_std)

        plt.grid(which='major',color='k',linestyle='--')
        plt.show()


    def get_lrf_data(self):
        return {
            'lrf_ave_steps':   self.lrf_ave_steps,
            'lrf_ave_samples': self.lrf_ave_samples,
            'lrs':             self.lrs,
            'losses':          self.losses,
            'losses_std':      self.losses_std,
            'lrf_x_range_min': self.lrf_x_range_min,
            'lrf_x_range_max': self.lrf_x_range_max,
        }

    def save_lrf(self, label, verbose=0, **kwargs):
        lrf = self.get_lrf_data()

        joblib.dump(lrf, label+'.dump.gz')

        if verbose:
            print('lrf data : ' + label+'.dump.gz')
            plot_lrf(lrf, **kwargs)




    def fit(
        self,
        trn_iter,
        val_iter=None,
        epochs=1,
        lrs=1.0e-3,
        wds=1e-7,
        wd_type=0,
        clip=1,
        onecycle_lr_fun=OneCycleGenerator(
            [0., 0.25, 1.],
            [0.1, 1.0, 0.01],
            interp_fun=interp_cos,
        ),
        onecycle_mom_fun=OneCycleGenerator(
            [0., 0.25, 1.],
            [0.95, 0.85, 0.95],
            interp_fun=interp_cos,
        ),
        grad_acc=1,
        teacher_forcing=None,
        mixup=False,
        mixup_alpha=1.0,
        mixup_max_lam=False,
        verbose=0):
        """Fit the model.

        Parameters
        ----------
        trn_iter : Iterator
            Data loader for training, providing input data andthe labels.
        val_iter : Iterator (optional)
            Data loader for validation, providing input data andthe labels.
        epoch : Int (optional)
            Number of epochs for traning.
            The default value is 1.
        lrs : Float or list/tuple/np.array of Floats (optional)
            Learning rate for training. If `onecycle_lr_fun` is given,
            this is the maximum (peak) value for the cycle.
            If list/tuple/np.array is given, the length must be equal to
            the number of layer groups of the model and each value
            corresponds to the layer group.
            The default value is 1.0e-3.
        wds : Float or list/tuple/np.array of Floats (optional)
            Weight decay factor.
            If list/tuple/np.array is given, the length must be equal to
            the number of layer groups of the model and each value
            corresponds to the layer group.
            The default value is 1e-7.
        wd_type : Int (optional)
            If `0` is given, AdamW type weight decay is used.
            Otherwise, conventional L2 regularization type is used.
        clip: None or Float (optional)
            Gradient clipping value.
            If None is given, gradient clipping is not applied.
            Default is 1.
        onecycle_lr_fun : function or None
            Function for hyper-parameter scheduling for learning rate.
            This function accepts one argument `pos` in the range `[0, 1]`
            and returns the value of the reference learning rate
            (that of the final layer group). The parameter `pos` is
            the relative position of the entire training steps.
        onecycle_mom_fun : function or None
            Function for hyper-parameter scheduling for momentum.
            This function accepts one argument `pos` in the range `[0, 1]`
            and returns the value of the momentum of the optimizer.
            The parameter `pos` is the relative position of
            the entire training steps.
        grad_acc : Int
            Gradient accumulation factor.
            For example, if `2` is given, updating parameters are applied
            once in 2 steps, so that effective batch size becomes twice.
            Default is 1.
        teacher_forcing : None or Int
            This is only for RNN models.
            Number of epochs for which teacher forcing is used.
            Default is `None`.
        mixup : bool
            Set `True` to use mixup data augmentation.
        mixup_alpha : Float (optional)
            Mixup parameter `alpha`. Only effective if `mixup=True`.
        mixup_max_lam : bool
            Set `True` to apply `max(lam, 1-lam)` for the parameter `lam`.
        verbose : Int
            0 prints nothing.
            1 prints current loss at each epoch.
            2 also prints progress bars.
            3 for google colaboratory.

        Returns
        -------
        h_metrics : numpy.array
            Histories of the training process are returned.
        """

        lrs0, wds0 = self._normalize_hyperparameters(lrs, wds)
        if wd_type == 0:
            # AdamW
            optimizer = self._init_optimizer_(lrs0)
        else:
            # L2 regularization
            optimizer = self._init_optimizer(lrs0, wds0)


        b_val = True if not val_iter is None else False

        cur_best_weights = copy.deepcopy(self.model.state_dict())
        cur_best_loss = None

        h_metrics = {k: [] for k in self.loss_dict.keys()}
        if b_val:
            for k in self.loss_dict.keys(): h_metrics['val_' + k] = []

        loss_all = []
        acc_all = []


        # Teacher Forcing
        tf_epochs = 0
        if not teacher_forcing is None:
            tf_epochs = teacher_forcing

        # One-Cycle LR scheduling
        if not onecycle_mom_fun is None:
            onecycle_nb = len(trn_iter) * epochs // grad_acc

            is_betas_in_optim = False
            onecycle_betas0 = None
            for pg in optimizer.param_groups:
                if 'betas' in pg:
                    is_betas_in_optim = True
                    onecycle_betas0 = pg['betas']
                    break


        _lrs = []
        _moms = []

        # initialize lr for the first training
        # One-Cycle LR scheduling
        if not onecycle_lr_fun is None:
            onecycle_lr_ratio = onecycle_lr_fun(0)
        else:
            onecycle_lr_ratio = 1
        if not onecycle_mom_fun is None:
            # mom is the absolute value of mom
            # mom0 is the initial value of mom
            # (mom/mom0) is gonna be the ratio
            mom = mom0 = onecycle_mom_fun(0)

        for lr0, pg in zip(lrs0, optimizer.param_groups):
            pg['lr'] = lr0 * onecycle_lr_ratio
            if not onecycle_mom_fun is None:
                if is_betas_in_optim:
                    pg['betas'] = [b0 for b0 in onecycle_betas0]
                else:
                    pg['momentum'] = mom


        # loop for epochs
        if verbose == 2:
            iter_epochs = tqdm_notebook(
                range(epochs), desc='epoch')
        else:
            iter_epochs = range(epochs)
        for ep in iter_epochs:

            # teacher forcing
            if not teacher_forcing is None:
                self.model[1].pr_force = (tf_epochs - ep) * 0.1 if ep < tf_epochs else 0

            n_trn = 0
            n_val = 0
            trn_metrics = {k: [] for k in self.loss_dict.keys()}
            if b_val: val_metrics = {k: [] for k in self.loss_dict.keys()}
            av_trn_metrics = {k: 0.0 for k in self.loss_dict.keys()}
            if b_val: av_val_metrics = {k: 0.0 for k in self.loss_dict.keys()}

            # training loop
            self.model.train(mode=True)

            if verbose == 3:
                # colab mode
                desc = 'epoch %3d train batch' % (ep)
                iter_train_steps = tqdm(trn_iter, desc=desc)
            elif verbose == 2:
                iter_train_steps = tqdm_notebook(
                    trn_iter, desc='train batch:', leave=False)
            else:
                iter_train_steps = trn_iter


            # zero the parameter gradients
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
            loss_acc = 0
            n1_trn = 0
            av1_trn_metrics = {k: 0.0 for k in self.loss_dict.keys()}
            if b_val: av1_val_metrics = {k: 0.0 for k in self.loss_dict.keys()}

            for ga_i, (inputs, labels) in enumerate(iter_train_steps):
                #inputs = inputs.to(self.device)
                #labels = labels.to(self.device)
                if isinstance(inputs, (list, tuple)):
                    inputs = [i.to(self.device) for i in inputs]
                else:
                    inputs = inputs.to(self.device)
                if isinstance(labels, (list, tuple)):
                    labels = [l.to(self.device) for l in labels]
                else:
                    labels = labels.to(self.device)

                if mixup:
                    inputs, labels_a, labels_b, lam = mixup_data(
                        inputs, labels, self.device,
                        mixup_alpha, mixup_max_lam)

                # forward
                with torch.set_grad_enabled(True):
                    # teacher forcing
                    if not teacher_forcing is None:
                        outputs = self.model((inputs, labels))
                    else:
                        outputs = self.model(inputs)

                    if self.is_rnn and isinstance(outputs, tuple):
                        outputs, *xtra = outputs

                    if mixup:
                        loss = mixup_criterion(
                            self.loss_dict['loss'], outputs,
                            labels_a, labels_b, lam)
                    else:
                        loss = self.loss_dict['loss'](outputs, labels)
                    raw_loss_ = loss.item()

                    if self.is_rnn and not self.reg_func is None:
                        loss = self.reg_func(outputs, xtra, loss)

                    loss = loss / grad_acc

                    # backward -- accumulating the gradients
                    loss.backward()

                loss_acc1 = raw_loss_ / grad_acc

                n1_trn += inputs.size()[0]
                trn_metrics['loss'].append(loss_acc1)
                av1_trn_metrics['loss'] += loss_acc1
                for k, m_fn in self.loss_dict.items():
                    if k != 'loss':
                        if mixup:
                            metric = mixup_criterion(
                                m_fn, outputs,
                                labels_a, labels_b, lam)
                        else:
                            metric = m_fn(outputs, labels)

                        trn_metrics[k].append(metric)
                        av1_trn_metrics[k] += metric

                loss_acc += loss_acc1
                if (ga_i + 1) % grad_acc == 0:
                    # finish a batch

                    with torch.set_grad_enabled(True):
                        # Gradient clipping
                        if clip:
                            nn.utils.clip_grad_norm_(
                                _trainable_params(self.model), clip)

                        # weight decay
                        if wd_type == 0:
                            # AdamW
                            for wd, pg in zip(wds0, optimizer.param_groups):
                                for p in pg['params']:
                                    p.data = p.data.add(-wd * pg['lr'], p.data)

                        # update the parameters
                        optimizer.step()

                        # zero the parameter gradients
                        optimizer.zero_grad()

                    n_trn += n1_trn
                    av_trn_metrics['loss'] += av1_trn_metrics['loss'] * n1_trn
                    for k, m_fn in self.loss_dict.items():
                        if k != 'loss':
                            av_trn_metrics[k] += av1_trn_metrics[k] * n1_trn

                    loss_acc = 0
                    n1_trn = 0
                    av1_trn_metrics = {k: 0.0 for k in self.loss_dict.keys()}
                    if b_val: av1_val_metrics = {k: 0.0 for k in self.loss_dict.keys()}

                    # One-Cycle LR scheduling
                    # house-keeping and
                    # seting new lr for the next training
                    if not onecycle_lr_fun is None:
                        _lrs.append(lrs0[-1] * onecycle_lr_ratio)
                        if not onecycle_mom_fun is None:
                            if is_betas_in_optim:
                                _moms.append(onecycle_betas0[0] * mom/mom0)
                            else:
                                _moms.append(mom)

                        t = (ep * len(trn_iter) + ga_i) / (epochs * len(trn_iter))
                        onecycle_lr_ratio = onecycle_lr_fun(t)
                        if not onecycle_mom_fun is None:
                            mom = onecycle_mom_fun(t)

                        for lr0, pg in zip(lrs0, optimizer.param_groups):
                            pg['lr'] = lr0 * onecycle_lr_ratio
                            if not onecycle_mom_fun is None:
                                if is_betas_in_optim:
                                    pg['betas'] = [
                                        b0 * mom/mom0 for b0 in onecycle_betas0
                                    ]
                                else:
                                    pg['momentum'] = mom

                    else:
                        # no One-Cycle LR scheduling
                        _lrs.append(lrs0[-1])


            # validation loop
            if b_val:
                # Set self.model to evaluate mode
                self.model.train(mode=False)
                self.model.eval()
    
                if verbose == 3:
                    desc = 'epoch %3d valid batch' % (ep)
                    iter_val_steps = tqdm(val_iter, desc=desc)
                elif verbose == 2:
                    iter_val_steps = tqdm_notebook(
                        val_iter, desc='val batch:', leave=False)
                else:
                    iter_val_steps = val_iter
    
                for inputs, labels in iter_val_steps:
                    #inputs = inputs.to(self.device)
                    #labels = labels.to(self.device)
                    if isinstance(inputs, (list, tuple)):
                        inputs = [i.to(self.device) for i in inputs]
                    else:
                        inputs = inputs.to(self.device)
                    if isinstance(labels, (list, tuple)):
                        labels = [l.to(self.device) for l in labels]
                    else:
                        labels = labels.to(self.device)
    
                    with torch.set_grad_enabled(False):
                        outputs = self.model(inputs)
                        if self.is_rnn and isinstance(outputs, tuple):
                            outputs, *xtra = outputs
    
                        loss = self.loss_dict['loss'](outputs, labels)

                    n1_val = inputs.size()[0]
                    n_val += n1_val
                    _loss = loss.item()
                    val_metrics['loss'].append(_loss)
                    av_val_metrics['loss'] += (_loss * n1_val)
                    for k, m_fn in self.loss_dict.items():
                        if k != 'loss':
                            metric = m_fn(outputs, labels)
                            val_metrics[k].append(metric)
                            av_val_metrics[k] += (metric * n1_val)


            # end of the epoch
            for l in self.loss_dict.keys():
                h_metrics[l].append(av_trn_metrics[l] / n_trn)
    
            if b_val:
                for l in self.loss_dict.keys():
                    h_metrics['val_' + l].append(av_val_metrics[l] / n_val)

            # acculumate the details for loss and acc
            loss_all.extend(trn_metrics['loss'])
            if 'acc' in self.loss_dict:
                acc_all.extend(trn_metrics['acc'])

            # check best_loss
            if b_val:
                ave_val_loss = h_metrics['val_loss'][-1]
                if cur_best_loss is None:
                    cur_best_loss = ave_val_loss
                elif ave_val_loss < cur_best_loss:
                    cur_best_loss = ave_val_loss
                    cur_best_weights = copy.deepcopy(self.model.state_dict())
            else:
                ave_loss = h_metrics['loss'][-1]
                if cur_best_loss is None:
                    cur_best_loss = ave_loss
                elif ave_loss < cur_best_loss:
                    cur_best_loss = ave_loss
                    cur_best_weights = copy.deepcopy(self.model.state_dict())
    
    
            if 'acc' in self.loss_dict:
                if verbose == 3:
                    if b_val:
                        tqdm.write('\nepoch %3d : (loss, val_loss, acc, val_acc)  %10f  %10f  %10f  %10f' % (
                            ep,
                            h_metrics['loss'][-1], h_metrics['val_loss'][-1],
                            h_metrics['acc'][-1], h_metrics['val_acc'][-1]))
                    else:
                        tqdm.write('\nepoch %3d : (loss, acc)  %10f  %10f' % (
                            ep,
                            h_metrics['loss'][-1],
                            h_metrics['acc'][-1]))
                elif verbose != 0:
                    if b_val:
                        tqdm.write('epoch %3d : (loss, val_loss, acc, val_acc)  %10f  %10f  %10f  %10f' % (
                            ep,
                            h_metrics['loss'][-1], h_metrics['val_loss'][-1],
                            h_metrics['acc'][-1], h_metrics['val_acc'][-1]))
                    else:
                        tqdm.write('epoch %3d : (loss, acc)  %10f  %10f' % (
                            ep,
                            h_metrics['loss'][-1],
                            h_metrics['acc'][-1]))
            else:
                if verbose == 3:
                    if b_val:
                        tqdm.write('\nepoch %3d : (loss, val_loss)  %10f  %10f' % (
                            ep,
                            h_metrics['loss'][-1], h_metrics['val_loss'][-1]))
                    else:
                        tqdm.write('\nepoch %3d : (loss)  %10f' % (
                            ep,
                            h_metrics['loss'][-1]))
                elif verbose != 0:
                    if b_val:
                        tqdm.write('epoch %3d : (loss, val_loss)  %10f / %10f' % (
                            ep,
                            h_metrics['loss'][-1], h_metrics['val_loss'][-1]))
                    else:
                        tqdm.write('epoch %3d : (loss)  %10f' % (
                            ep,
                            h_metrics['loss'][-1]))
    
    
        self.best_weights = cur_best_weights
        self.best_loss = cur_best_loss
    
        h_metrics['lrs'] = _lrs
        if not onecycle_mom_fun is None:
            h_metrics['moms'] = _moms
        h_metrics['loss_all'] = loss_all
        if 'acc' in self.loss_dict:
            h_metrics['acc_all'] = acc_all
        return h_metrics


    def evaluate(
        self,
        val_iter,
        eval_func,
        verbose=0):
        """Evaluate the model for an input data and return the predictions
        and the corresponding labels.

        Parameters
        ----------
        val_iter : Iterator
            Data loader providing input data as well as the labels.
        eval_func: Function
            Function like eval_func(outputs, labels) which returns
            preds and labels.
        verbose: Int
            Set non zero value for verbose mode.

        Returns
        -------
        all_preds : numpy.array
            Predictions of the model for all samples of input data.
        all_labels : numpy.array
            The ground truth labels for all samples of input data.
        """

        all_preds = None
        all_labels = None

        # Set self.model to evaluate mode
        self.model.train(mode=False)
        self.model.eval()

        if verbose == 3:
            iter_val_steps = tqdm(val_iter, desc='valid batch')
        elif verbose == 2:
            iter_val_steps = tqdm_notebook(
                val_iter, desc='val batch:', leave=False)
        else:
            iter_val_steps = val_iter

        for inputs, labels in iter_val_steps:
            #inputs = inputs.to(self.device)
            #labels = labels.to(self.device)
            if isinstance(inputs, (list, tuple)):
                inputs = [i.to(self.device) for i in inputs]
            else:
                inputs = inputs.to(self.device)
            if isinstance(labels, (list, tuple)):
                labels = [l.to(self.device) for l in labels]
            else:
                labels = labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                if self.is_rnn and isinstance(outputs, tuple):
                    outputs, *xtra = outputs

                preds, labels = eval_func(outputs, labels)

            #np_preds = preds.numpy()
            #np_labels = np.array(labels)
            np_preds = preds.cpu().data.numpy()
            np_labels = labels.detach().cpu().numpy()

            if all_preds is None: all_preds = np_preds
            else: all_preds = np.concatenate((all_preds, np_preds))

            if all_labels is None: all_labels = np_labels
            else: all_labels = np.concatenate((all_labels, np_labels))

        return all_preds, all_labels


    def eval_TTA(
        self,
        val_iter,
        tta=1,
        model_output='log_softmax',
        verbose=0):
        """Evaluate the model for an input data and return the predictions
        and the corresponding labels with TTA (test time augmentation).

        Parameters
        ----------
        val_iter : Iterator
            Data loader providing input data as well as the labels.
            Shuffling must be turned off.
        tta : Int
            Number of sampling for TTA.
        model_output : Str
            The type of model output, 'log_softmax', 'softmax', 'sigmoid'
            and 'raw'. Default if 'log_softmax'.
        verbose: Int
            Set non zero value for verbose mode.

        Returns
        -------
        all_preds : numpy.array
            Predictions of the model for all samples of input data.
        all_labels : numpy.array
            The ground truth labels for all samples of input data.
        """

        ave_outputs = None

        # Set self.model to evaluate mode
        self.model.train(mode=False)
        self.model.eval()

        # loop for ttas
        if verbose == 2:
            iter_ttas = tqdm_notebook(
                range(tta), desc='epoch:')
        else:
            iter_ttas = range(tta)
        for ep in iter_ttas:

            all_outputs = None
            all_labels = None

            if verbose == 3:
                desc = 'epoch %3d valid batch' % (ep)
                iter_val_steps = tqdm(val_iter, desc=desc)
            elif verbose == 2:
                iter_val_steps = tqdm_notebook(
                    val_iter, desc='val batch:', leave=False)
            else:
                iter_val_steps = val_iter

            for inputs, labels in iter_val_steps:
                #inputs = inputs.to(self.device)
                #labels = labels.to(self.device)
                if isinstance(inputs, (list, tuple)):
                    inputs = [i.to(self.device) for i in inputs]
                else:
                    inputs = inputs.to(self.device)
                if isinstance(labels, (list, tuple)):
                    labels = [l.to(self.device) for l in labels]
                else:
                    labels = labels.to(self.device)

                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)
                    if self.is_rnn and isinstance(outputs, tuple):
                        outputs, *xtra = outputs

                #new_outputs = outputs.detach().cpu().numpy()
                if model_output == 'log_softmax':
                    new_outputs = np.exp(outputs.detach().cpu().numpy())
                elif model_output in ['softmax', 'sigmoid']:
                    new_outputs = outputs.detach().cpu().numpy()
                else:
                    new_outputs = F.softmax(outputs, dim=-1).detach().cpu().numpy()

                if all_outputs is None:
                    all_outputs = new_outputs
                else:
                    all_outputs = np.concatenate(
                        (all_outputs, new_outputs)
                    )

                #if all_labels is None:
                #    all_labels = np.array(labels)
                #else:
                #    all_labels = np.concatenate(
                #        (all_labels,
                #         np.array(labels)))
                np_labels = labels.detach().cpu().numpy()
                if all_labels is None: all_labels = np_labels
                else: all_labels = np.concatenate((all_labels, np_labels))

            if ave_outputs is None:
                ave_outputs = all_outputs
            else:
                ave_outputs = ave_outputs + all_outputs

        # argmax is invaliant with the scaling
        return np.argmax(ave_outputs, axis=1), all_labels


    def eval_TTA_(
        self,
        val_iter,
        eval_func,
        tta=1,
        model_output='log_softmax',
        verbose=0):
        """Evaluate the model for an input data and return the predictions
        and the corresponding labels with TTA (test time augmentation).

        Parameters
        ----------
        val_iter : Iterator
            Data loader providing input data as well as the labels.
            Shuffling must be turned off.
        eval_func: Function
            Function like eval_func(outputs, labels) which returns
            preds and labels.
        tta : Int
            Number of sampling for TTA.
        model_output : Str
            The type of model output, 'log_softmax', 'softmax', 'sigmoid'
            and 'raw'. Default if 'log_softmax'.
        verbose: Int
            Set non zero value for verbose mode.

        Returns
        -------
        all_preds : numpy.array
            Predictions of the model for all samples of input data.
        all_labels : numpy.array
            The ground truth labels for all samples of input data.
        """

        ave_outputs = None

        # Set self.model to evaluate mode
        self.model.train(mode=False)
        self.model.eval()

        # loop for ttas
        if verbose == 2:
            iter_ttas = tqdm_notebook(
                range(tta), desc='epoch:')
        else:
            iter_ttas = range(tta)
        for ep in iter_ttas:

            all_outputs = None
            all_labels = None

            if verbose == 3:
                desc = 'epoch %3d valid batch' % (ep)
                iter_val_steps = tqdm(val_iter, desc=desc)
            elif verbose == 2:
                iter_val_steps = tqdm_notebook(
                    val_iter, desc='val batch:', leave=False)
            else:
                iter_val_steps = val_iter

            #for inputs, labels in iter_val_steps:
            for data in iter_val_steps:
                if len(data) == 2:
                    inputs, labels = data
                elif len(data) == 1:
                    inputs = data
                    labels = None
                else:
                    raise ValueError('Data loader `val_iter` must return `(inputs, labels)` or `inputs`')

                #inputs = inputs.to(self.device)
                if isinstance(inputs, (list, tuple)):
                    inputs = [i.to(self.device) for i in inputs]
                else:
                    inputs = inputs.to(self.device)

                if not labels is None:
                    #labels = labels.to(self.device)
                    if isinstance(labels, (list, tuple)):
                        labels = [l.to(self.device) for l in labels]
                    else:
                        labels = labels.to(self.device)

                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)
                    if self.is_rnn and isinstance(outputs, tuple):
                        outputs, *xtra = outputs

                outputs, labels = eval_func(outputs, labels)
                new_outputs = outputs.detach().cpu().numpy()

                if all_outputs is None:
                    all_outputs = new_outputs
                else:
                    all_outputs = np.concatenate(
                        (all_outputs, new_outputs)
                    )

                if not labels is None:
                    if all_labels is None:
                        all_labels = np.array(labels)
                    else:
                        all_labels = np.concatenate(
                            (all_labels,
                             np.array(labels)))

            if ave_outputs is None:
                ave_outputs = all_outputs
            else:
                ave_outputs = ave_outputs + all_outputs

        return ave_outputs/tta, all_labels


    def save_history(self, res, label, verbose=0):
        # save res (fitting history) to dump file
        joblib.dump(res, label+'.dump.gz')

        # save weights
        self.save_weights(label+'_weights.pth', best=False)
        self.save_weights(label+'_best_weights.pth', best=True)

        if verbose != 0:
            print('history      : ' + label+'.dump.gz')
            print('weights      : ' + label+'_weights.pth')
            print('best weights : ' + label+'_best_weights.pth')
            print('best loss    :', self.best_loss)
            plot_history(res)


    def save_weights(self, filename, best=True):
        """Save model weights. This is a wrapper of torch.save().

        Parameters
        ----------
        filename : string
            File path to the output file.
        best : bool (optional)
            When True, self.best_weights is saved.
            Otherwise, the weights of the current model
            obtained by self.model.state_dict() is saved.
        """

        if os.path.exists(filename):
            print('%s already exists.' % (filename))
            print('file does not saved.')
            return

        if best:
            if self.best_weights is None:
                print('best_weights is None.')
                return
            state_dict = self.best_weights
        else:
            state_dict = self.model.state_dict()

        torch.save(state_dict, filename)


    def load_weights(self, filename):
        """Load model weights from file saved by torch.save().
        Note that the weights are loaded to the current model
        and the property self.best_weights does not changed.

        Parameters
        ----------
        filename : string
            File path to the model weights file.
        best : bool (optional)
            When True, self.best_weights is saved.
            Otherwise, the weights of the current model
            obtained by self.model.state_dict() is saved.
        """

        if not os.path.exists(filename):
            print('%s does not exist.' % (filename))
            return
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)




class GAN():
    """GAN model class

    Parameters
    ----------
    netG : torch.nn.Module.
        Generator model
    netD : torch.nn.Module
        Discriminator model.
    nz : int
        Size of the input vector to the Generator.
    device: torch.device
        Set torch.device('cuda') or torch.device('cpu').
    optG : torch.optim
        Optimizer for the Generator.
    optD : torch.optim
        Optimizer for the Discriminator.
    criterion : 
        Loss function for the output of the Discriminator.
        Default is `nn.BCELoss()`.
    """

    def __init__(
        self,
        netG, netD, nz,
        device,
        optG, optD,
        criterion=nn.BCELoss()):

        self.netG = netG
        self.netD = netD
        self.nz = nz
        self.device = device
        self.optG = optG
        self.optD = optD
        self.criterion = criterion


    def fit(
        self,
        epochs,
        dataloader,
        iterG=1, iterD=5,
        real_label=1., fake_label=0.,
        fixed_noise=None,
        G_losses=[], D_losses=[], img_list=[], Ds=[],
        verbose=1,
        plot_images=True,
        n_images=8,
        figsize=(12, 12)):
        """Fit the GAN model

        Parameters
        ----------
        epoch : Int (optional)
            Number of epochs for traning.
            The default value is 1.
        dataloader : Iterator
            Dataloader for training, providing images and the corresponding labels.
        iterG :
            Iterations of generator training for each step.
            Default is 1.
        iterD :
            Iternations of discriminator training for each step.
            Default is 5.
        real_label :
            Label value for real images.
            Default is 1. If a value less than 1 is given, the label values are
            randomly sampled from the range of [real_label, 1] for each image.
        fake_label :
            Label value for fake images.
            Default is 0. If a value larger than 0 is given, the label values are
            randomly sampled from the range of [0, fake_label] for each image.
        fixed_noise :
            The noise vector to test the Generator.
            If `None` is given, one random vector is generated inside this method.
        G_losses :
            Array to record Generator losses in training. By giving this, you can
            record the new history after the prior training.
            Default is `[]`, that is, no prior history.
        D_losses :
            Array to record Discriminator losses in training.
            Default is `[]`.
        img_list :
            Array to record the generated images in training.
            Default is `[]`.
        Ds :
            Array to record the averaged Discriminator's results of real images,
            fake images and fake images at the Generator's training phasel
            Default is `[]`.
        verbose : Int (optional)
            If `0` is given, no information is printed during the training.
            If `1` is given, summaries of losses are printed at each epoch.
            If `2` is given, summaries of losses are printed at each step.
        plot_images : Bool (optional)
            If set `True`, generated images are plotted at each epoch.
            Number of images is specified by the option 'n_images'.
            Default is `True`.
        n_images : Int (optional)
            Numver of generated images plotted at each epoch
            if `plot_images == True`.
        figsize : list or tuple
            Figure size of the generated images plotted
            if `plot_images == True`.
        """

        if fixed_noise is None:
            fixed_noise = torch.randn(64, self.nz, 1, 1).to(self.device)

        n_loop = len(dataloader) // iterG // iterD
        dataiter = iter(dataloader)

    
        for epoch in range(epochs):
            for i_loop in range(n_loop):
                tmp_lossD = []
                tmp_lossG = []
                av_Dx = 0.
                av_DGz1 = 0.
                av_DGz2 = 0.

                # train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))

                # unfreeze D
                for p in self.netD.parameters():
                    p.requires_grad_(True)

                self.netD.zero_grad()
                for iD in range(iterD):

                    ## Train with all-real batch
                    data = next(dataiter, None)
                    if data is None:
                        dataiter = iter(dataloader)
                        data = next(dataiter)
                    b_size = data[0].size(0)
                    if n_images > b_size: n_images = b_size

                    if real_label < 1.0:
                        label = torch.FloatTensor(b_size).uniform_(real_label, 1.0).to(self.device)
                    else:
                        label = torch.full((b_size, ), real_label).to(self.device)

                    output = self.netD(data[0].to(self.device)).view(-1)
                    lossD_real = self.criterion(output, label)
                    lossD_real.backward()

                    av_Dx += output.detach().mean().item()

                    ## Train with all-fake batch
                    if fake_label > 0.0:
                        label.uniform_(0.0, fake_label)
                    else:
                        label.fill_(fake_label)

                    with torch.no_grad():
                        noise = torch.randn(b_size, self.nz, 1, 1).to(self.device)
                        fake = self.netG(noise)

                    output = self.netD(fake.detach()).view(-1)
                    lossD_fake = self.criterion(output, label)
                    lossD_fake.backward()

                    av_DGz1 += output.detach().mean().item()
                    lossD = lossD_real.detach() + lossD_fake.detach()
                    tmp_lossD.append(lossD.item())

                # Update D
                self.optD.step()
                D_losses.append(np.array(tmp_lossD).mean())
                av_Dx /= iterD
                av_DGz1 /= iterD


                # train Generator: maximize log(D(G(z)))

                # freeze D
                for p in self.netD.parameters():
                    p.requires_grad_(False)

                self.netG.zero_grad()
                for iG in range(iterG):

                    if real_label < 1.0:
                        label.uniform_(real_label, 1.0)
                    else:
                        label.fill_(real_label)

                    noise = torch.randn(b_size, self.nz, 1, 1).to(self.device)
                    fake = self.netG(noise)
                    output = self.netD(fake).view(-1)
                    lossG = self.criterion(output, label)
                    lossG.backward()

                    av_DGz2 += output.detach().mean().item()
                    tmp_lossG.append(lossG.detach().item())

                # Update G
                self.optG.step()
                G_losses.append(np.array(tmp_lossG).mean())
                av_DGz2 /= iterG

                Ds.append((av_Dx, av_DGz1, av_DGz2))

                if verbose > 1:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, epochs, i_loop, n_loop,
                             D_losses[-1], G_losses[-1], av_Dx, av_DGz1, av_DGz2))

            # Output training stats
            if verbose > 0:
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs,
                     np.array(G_losses)[-n_loop:].mean(),
                     np.array(D_losses)[-n_loop:].mean(),
                     np.array(Ds)[-n_loop:, 0].mean(),
                     np.array(Ds)[-n_loop:, 1].mean(),
                     np.array(Ds)[-n_loop:, 2].mean()))

            # Check how the generator is doing by saving G's output on fixed_noise
            with torch.no_grad():
                fake = self.netG(fixed_noise).detach().cpu()
            img = vutils.make_grid(fake[:n_images], padding=2, normalize=True)
            if plot_images:
                plt.figure(figsize=figsize)
                plt.imshow(img.numpy().transpose(1, 2, 0))
                plt.axis('off')
                plt.show()
            img_list.append(img)

        return G_losses, D_losses, img_list, Ds


    def save(self, history, filename):
        if not history is None:
            joblib.dump(
                history,
                filename + '.dump.gz')
        torch.save(
            self.netG.state_dict(),
            filename + '-netG_weights.pth')
        torch.save(
            self.netD.state_dict(),
            filename + '-netD_weights.pth')

        if not history is None:
            print('histories   : %s.dump.gz'%(filename))
        print('netG weights: %s-netG_weights.pth'%(filename))
        print('netD weights: %s-netD_weights.pth'%(filename))


    # returned value: history = (G_losses, D_losses, img_list)
    def load(self, filename):
        netG_file = filename + '-netG_weights.pth'
        netD_file = filename + '-netD_weights.pth'
        dump_file = filename + '.dump.gz'

        if os.path.exists(netG_file):
            self.netG.load_state_dict(torch.load(netG_file))
            print(f'{netG_file} is loaded')
        else:
            print('failed to open netG weights:', netG_file)

        if os.path.exists(netD_file):
            self.netD.load_state_dict(torch.load(netD_file))
            print(f'{netD_file} is loaded')
        else:
            print('failed to open netD weights:', netD_file)

        if os.path.exists(dump_file):
            return joblib.load(dump_file)

    @staticmethod
    def plot_losses(history, figsize=(16, 10)):
        # assume that history[0] is G_losses and
        # history[1] is D_losses.

        plt.figure(figsize=figsize)
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(history[0], label="G")
        plt.plot(history[1], label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    
    @staticmethod
    def plot_images(dataloader, imgs, n_images=8, figsize=(16, 16)):
        """
        example) plot_images(val_dl, h01[2][-1])
        this shows real images from `val_dl`
        and the generated images at the last epoch in `h01`.
        """

        plt.figure(figsize=figsize)

        # Grab a batch of real images from the dataloader
        real_batch = next(iter(dataloader))

        # Plot the real images
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    real_batch[0][:n_images],
                    padding=5, normalize=True),
                (1, 2, 0)
            )
        )

        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(imgs, (1, 2, 0)))

        plt.show()




class RCNN(model):
    """RCNN model class

    Parameters
    ----------
    model : torch.nn.Module
        The model to train, predict, etc.
    device: torch.device
        Set torch.device('cuda') or torch.device('cpu')
    opt_func: torch.optim
        The optimizer class for training. Note that this is NOT an instance.
    loss_dict: dictionary of the label (key) and the loss or metric function.
        For example, {'loss': F.cross_entropy, 'acc': accuracy}.
        The entry 'loss' is mandatory.
    layer_groups : None or list/tuple
        Layer groups for which different LRs and/or momentums can be set.
    is_rnn : bool
        Set `True` for the RNN model which outputs extra elements.
    reg_seq2seq : bool
        This is only for RNN models.
        If `True`, use regularization `reg_seq2seq`.
        Default is `False`.
    """

    def find_lr(
        self,
        trn_iter,
        ave_steps=1,
        lr_steps=10,
        lr_factor=1.1,
        max_lr=None,
        lrs=1.0e-6,
        wds=1e-7,
        wd_type=0,
        clip=1,
        grad_acc=1,
        flag_gc=False,
        verbose=0):
        """Find learning rate parameter.

        Parameters
        ----------
        trn_iter:
            Data loader (iterator) providing input data as well as the labels.
        ave_steps: Int
            Number of runs to take ensemble average of the results.
            If 0 is given, losses are evalulated in a continuous mode,
            where the weights are not reset for each `lr`.
            Default is 1, that is, no ensemble averaging.
        lr_steps: Int (optional)
            Steps to optimize for each lr value. The default value is 10.
        lr_factor: Float (optional)
            The multiplicative factor for the learning rate
            for the next cycle. The default value is 1.1.
        max_lr: Float (optional)
            The maximum lr value to search.
            If None is given, searching is terminated by the loss limit
            which is 4 times bigger than the initial loss value.
            Default is None.
        lrs : Float or list/tuple/np.array of Floats (optional)
            The initial value of the learning rates for LR finder.
            If list/tuple/np.array is given, the length must be equal to
            the number of layer groups of the model and each value
            corresponds to the layer group.
            Default is 1.0e-6.
        wds : Float or list/tuple/np.array of Floats (optional)
            Weight decay factor.
            If list/tuple/np.array is given, the length must be equal to
            the number of layer groups of the model and each value
            corresponds to the layer group.
            The default value is 1e-7.
        wd_type : Int (optional)
            If `0` is given, AdamW type weight decay is used.
            Otherwise, conventional L2 regularization type is used.
        clip: None or Float (optional)
            Gradient clipping value.
            If None is given, gradient clipping is not applied.
            Default is None.
        grad_acc : Int
            Gradient accumulation factor.
            For example, if `2` is given, updating parameters are applied
            once in 2 steps, so that effective batch size becomes twice.
            Default is 1.
        flag_gc : Bool (optional)
            Set `True` if you want gabage collecting for every step.
            Default is `False`.
        verbose: Int
            Set non zero value for verbose mode.
        """

        # for house keeping
        init_state_dict = copy.deepcopy(self.model.state_dict())

        continuous_mode = False
        if ave_steps == 0:
            continuous_mode = True
            ave_steps = 1

        self.lrf_ave_steps = ave_steps
        self.lrf_ave_samples = []

        average_count = 0
        _lrs = []
        _losses = []
        lr_ratio = 1 / lr_factor # to begin with lr_ratio = 1
        init_loss = None

        gen = iter(trn_iter)
        loss_func = self.loss_dict['loss']

        lrs0, wds0 = self._normalize_hyperparameters(lrs, wds)
        if wd_type == 0:
            # AdamW
            optimizer = self._init_optimizer_(lrs0)
        else:
            # L2 regularization
            optimizer = self._init_optimizer(lrs0, wds0)


        if max_lr is None: lr_range = None
        else: lr_range = max_lr / lrs0[-1]


        # training loop
        self.model.train(mode=True)

        while 1:
            lr_ratio *= lr_factor

            for lr0, pg in zip(lrs0, optimizer.param_groups):
                pg['lr'] = lr0 * lr_ratio

            if hasattr(self.model, 'reset'):
                self.model.reset()

            # training loop with the learning rate
            if not continuous_mode:
                self.model.load_state_dict(init_state_dict)
            for loop in range(lr_steps):

                # zero the parameter gradients
                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                loss_acc = 0

                # gradient accumulation loop
                for ga_i in range(grad_acc):
                    try:
                        inputs, labels = next(gen)
                    except StopIteration:
                        gen = iter(trn_iter)
                        inputs, labels = next(gen)

                    #inputs = inputs.to(self.device)
                    #labels = labels.to(self.device)
                    inputs = [i.to(self.device) for i in inputs]
                    labels = [{k: v.to(self.device) for k, v in l.items()} for l in labels]

                    # forward
                    with torch.set_grad_enabled(True):
                        #outputs = self.model(inputs)
                        # for RCNN models
                        outputs = self.model(inputs, labels)

                        loss = loss_func(outputs, labels)
                        raw_loss_ = loss.item()

                        loss = loss / grad_acc

                        # backward -- acculumating the grads
                        loss.backward()

                        # for OOM by Giang
                        if flag_gc:
                            del inputs, labels, outputs, loss
                            gc.collect()
                            torch.cuda.empty_cache()


                    loss_acc += raw_loss_ / grad_acc

                # update parameters
                with torch.set_grad_enabled(True):
                    # Gradient clipping
                    if clip:
                        nn.utils.clip_grad_norm_(
                            _trainable_params(self.model), clip)

                    # weight decay
                    if wd_type == 0:
                        # AdamW
                        for wd, pg in zip(wds0, optimizer.param_groups):
                            for p in pg['params']:
                                p.data = p.data.add(-wd * pg['lr'], p.data)

                    # zero the parameter gradients
                    optimizer.step()


            # the last loss (after lr_steps)
            cur_loss = loss_acc
            if init_loss is None: init_loss = cur_loss

            # representative lr
            lr = lr_ratio * lrs0[-1]

            _lrs.append(lr)
            _losses.append(cur_loss)

            if verbose != 0:
                print('\rave_count: %d, lr: %e, loss: %f'\
                    % (average_count, lr, cur_loss), end='')

            if math.isinf(cur_loss) or math.isnan(cur_loss) or\
               (lr_range is None and cur_loss > 4.0 * init_loss) or\
               (not lr_range is None and lr_ratio > lr_range):

                self.lrf_ave_samples.append((_lrs, _losses))

                # house keeping
                self.model.load_state_dict(init_state_dict)

                average_count += 1
                if average_count >= ave_steps:
                    # averaging process
                    self._lrf_finalize()
                    return

                # for restart
                _lrs = []
                _losses = []
                lr_ratio = 1.0
                init_loss = None


    def fit(
        self,
        trn_iter,
        val_iter=None,
        epochs=1,
        lrs=1.0e-3,
        wds=1e-7,
        wd_type=0,
        clip=1,
        onecycle_lr_fun=OneCycleGenerator(
            [0., 0.25, 1.],
            [0.1, 1.0, 0.01],
            interp_fun=interp_cos,
        ),
        onecycle_mom_fun=OneCycleGenerator(
            [0., 0.25, 1.],
            [0.95, 0.85, 0.95],
            interp_fun=interp_cos,
        ),
        grad_acc=1,
        flag_gc=False,
        verbose=0):
        """Fit the model.

        Parameters
        ----------
        trn_iter : Iterator
            Data loader for training, providing input data andthe labels.
        val_iter : Iterator (optional)
            Data loader for validation, providing input data andthe labels.
        epoch : Int (optional)
            Number of epochs for traning.
            The default value is 1.
        lrs : Float or list/tuple/np.array of Floats (optional)
            Learning rate for training. If `use_clr` is `True`,
            this is the maximum (peak) value for the cycle.
            If list/tuple/np.array is given, the length must be equal to
            the number of layer groups of the model and each value
            corresponds to the layer group.
            The default value is 1.0e-3.
        wds : Float or list/tuple/np.array of Floats (optional)
            Weight decay factor.
            If list/tuple/np.array is given, the length must be equal to
            the number of layer groups of the model and each value
            corresponds to the layer group.
            The default value is 1e-7.
        wd_type : Int (optional)
            If `0` is given, AdamW type weight decay is used.
            Otherwise, conventional L2 regularization type is used.
        clip: None or Float (optional)
            Gradient clipping value.
            If None is given, gradient clipping is not applied.
            Default is 1.
        onecycle_lr_fun : function or None
            Function for hyper-parameter scheduling for learning rate.
            This function accepts one argument `pos` in the range `[0, 1]`
            and returns the value of the reference learning rate
            (that of the final layer group). The parameter `pos` is
            the relative position of the entire training steps.
        onecycle_mom_fun : function or None
            Function for hyper-parameter scheduling for momentum.
            This function accepts one argument `pos` in the range `[0, 1]`
            and returns the value of the momentum of the optimizer.
            The parameter `pos` is the relative position of
            the entire training steps.
        grad_acc : Int
            Gradient accumulation factor.
            For example, if `2` is given, updating parameters are applied
            once in 2 steps, so that effective batch size becomes twice.
            Default is 1.
        flag_gc : Bool (optional)
            Set `True` if you want gabage collecting for every step.
            Default is `False`.
        verbose : Int
            0 prints nothing.
            1 prints current loss at each epoch.
            2 also prints progress bars.
            3 for google colaboratory.

        Returns
        -------
        h_metrics : numpy.array
            Histories of the training process are returned.
        """

        lrs0, wds0 = self._normalize_hyperparameters(lrs, wds)
        if wd_type == 0:
            # AdamW
            optimizer = self._init_optimizer_(lrs0)
        else:
            # L2 regularization
            optimizer = self._init_optimizer(lrs0, wds0)


        b_val = True if not val_iter is None else False

        cur_best_weights = copy.deepcopy(self.model.state_dict())
        cur_best_loss = None

        h_metrics = {'loss': []}
        if b_val:
            for k in self.loss_dict.keys():
                if k == 'loss': continue
                h_metrics['val_' + k] = []

        loss_all = []


        # One-Cycle LR scheduling
        if not onecycle_mom_fun is None:
            onecycle_nb = len(trn_iter) * epochs // grad_acc

            is_betas_in_optim = False
            onecycle_betas0 = None
            for pg in optimizer.param_groups:
                if 'betas' in pg:
                    is_betas_in_optim = True
                    onecycle_betas0 = pg['betas']
                    break


        _lrs = []
        _moms = []

        # initialize lr for the first training
        # One-Cycle LR scheduling
        if not onecycle_lr_fun is None:
            onecycle_lr_ratio = onecycle_lr_fun(0)
        else:
            onecycle_lr_ratio = 1
        if not onecycle_mom_fun is None:
            # mom is the absolute value of mom
            # mom0 is the initial value of mom
            # (mom/mom0) is gonna be the ratio
            mom = mom0 = onecycle_mom_fun(0)

        for lr0, pg in zip(lrs0, optimizer.param_groups):
            pg['lr'] = lr0 * onecycle_lr_ratio
            if not onecycle_mom_fun is None:
                if is_betas_in_optim:
                    pg['betas'] = [b0 for b0 in onecycle_betas0]
                else:
                    pg['momentum'] = mom


        # loop for epochs
        if verbose == 2:
            iter_epochs = tqdm_notebook(
                range(epochs), desc='epoch')
        else:
            iter_epochs = range(epochs)
        for ep in iter_epochs:

            n_trn = 0; n_val = 0
            trn_metrics = {'loss': []}
            if b_val: val_metrics = {k: [] for k in self.loss_dict.keys() if k != 'loss'}

            # training loop
            self.model.train(mode=True)

            if verbose == 3:
                # colab mode
                desc = 'epoch %3d train batch' % (ep)
                iter_train_steps = tqdm(trn_iter, desc=desc)
            elif verbose == 2:
                iter_train_steps = tqdm_notebook(
                    trn_iter, desc='train batch:', leave=False)
            else:
                iter_train_steps = trn_iter


            # zero the parameter gradients
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
            loss_acc = 0

            for ga_i, (inputs, labels) in enumerate(iter_train_steps):
                #inputs = inputs.to(self.device)
                #labels = labels.to(self.device)
                inputs = [i.to(self.device) for i in inputs]
                labels = [{k: v.to(self.device) for k, v in l.items()} for l in labels]


                # forward
                with torch.set_grad_enabled(True):
                    #outputs = self.model(inputs)
                    # for RCNN models
                    outputs = self.model(inputs, labels)

                    loss = self.loss_dict['loss'](outputs, labels)
                    raw_loss_ = loss.item()

                    loss = loss / grad_acc

                    # backward -- accumulating the gradients
                    loss.backward()

                    # for OOM by Giang
                    if flag_gc:
                        del inputs, labels, outputs, loss
                        gc.collect()
                        torch.cuda.empty_cache()


                loss_acc += raw_loss_ / grad_acc
                if (ga_i + 1) % grad_acc == 0:
                    # finish a batch

                    with torch.set_grad_enabled(True):
                        # Gradient clipping
                        if clip:
                            nn.utils.clip_grad_norm_(
                                _trainable_params(self.model), clip)

                        # weight decay
                        if wd_type == 0:
                            # AdamW
                            for wd, pg in zip(wds0, optimizer.param_groups):
                                for p in pg['params']:
                                    p.data = p.data.add(-wd * pg['lr'], p.data)

                        # update the parameters
                        optimizer.step()

                        # zero the parameter gradients
                        optimizer.zero_grad()

                    n_trn += 1
                    trn_metrics['loss'].append(loss_acc)
                    # ... skip metrics other than loss in training for now...

                    loss_acc = 0

                    # One-Cycle LR scheduling
                    # house-keeping and
                    # seting new lr for the next training
                    if not onecycle_lr_fun is None:
                        _lrs.append(lrs0[-1] * onecycle_lr_ratio)
                        if not onecycle_mom_fun is None:
                            if is_betas_in_optim:
                                _moms.append(onecycle_betas0[0] * mom/mom0)
                            else:
                                _moms.append(mom)

                        t = (ep * len(trn_iter) + ga_i) / (epochs * len(trn_iter))
                        onecycle_lr_ratio = onecycle_lr_fun(t)
                        if not onecycle_mom_fun is None:
                            mom = onecycle_mom_fun(t)

                        for lr0, pg in zip(lrs0, optimizer.param_groups):
                            pg['lr'] = lr0 * onecycle_lr_ratio
                            if not onecycle_mom_fun is None:
                                if is_betas_in_optim:
                                    pg['betas'] = [
                                        b0 * mom/mom0 for b0 in onecycle_betas0
                                    ]
                                else:
                                    pg['momentum'] = mom

                    else:
                        # no One-Cycle LR scheduling
                        _lrs.append(lrs0[-1])


            # validation loop
            if b_val:
                # Set self.model to evaluate mode
                self.model.train(mode=False)
                self.model.eval()
                # to obtain loss

                if verbose == 3:
                    desc = 'epoch %3d valid batch' % (ep)
                    iter_val_steps = tqdm(val_iter, desc=desc)
                elif verbose == 2:
                    iter_val_steps = tqdm_notebook(
                        val_iter, desc='val batch:', leave=False)
                else:
                    iter_val_steps = val_iter
    
                for inputs, labels in iter_val_steps:
                    #inputs = inputs.to(self.device)
                    #labels = labels.to(self.device)
                    inputs = [i.to(self.device) for i in inputs]
                    labels = [{k: v.to(self.device) for k, v in l.items()} for l in labels]

                    with torch.set_grad_enabled(False):
                        outputs = self.model(inputs)
    
                    n_val += 1
                    for k, m_fn in self.loss_dict.items():
                        if k == 'loss': continue
                        val_metrics[k].append(m_fn(outputs, labels))

                    # for OOM by Giang
                    if flag_gc:
                        del inputs, labels, outputs
                        gc.collect()
                        torch.cuda.empty_cache()


            # end of the epoch
            h_metrics['loss'].append(np.array(trn_metrics['loss']).sum() / n_trn)
            if b_val:
                for l in self.loss_dict.keys():
                    if l == 'loss': continue
                    h_metrics['val_' + l].append(np.array(val_metrics[l]).sum() / n_val)

            # acculumate the details for loss and acc
            loss_all.extend(trn_metrics['loss'])

            # check best_loss
            # NOTE: use training loss because no val_loss
            ave_loss = h_metrics['loss'][-1]
            if cur_best_loss is None:
                cur_best_loss = ave_loss
            elif ave_loss < cur_best_loss:
                cur_best_loss = ave_loss
                cur_best_weights = copy.deepcopy(self.model.state_dict())
    
    
            if 'acc' in self.loss_dict:
                if verbose == 3:
                    if b_val:
                        tqdm.write('\nepoch %3d : (loss, val_acc)  %10f  %10f' % (
                            ep,
                            h_metrics['loss'][-1], h_metrics['val_acc'][-1]))
                    else:
                        tqdm.write('\nepoch %3d : (loss)  %10f' % (
                            ep,
                            h_metrics['loss'][-1]))
                elif verbose != 0:
                    if b_val:
                        tqdm.write('epoch %3d : (loss, val_acc)  %10f  %10f' % (
                            ep,
                            h_metrics['loss'][-1], h_metrics['val_acc'][-1]))
                    else:
                        tqdm.write('epoch %3d : (loss)  %10f' % (
                            ep,
                            h_metrics['loss'][-1]))
            else:
                if verbose == 3:
                    tqdm.write('\nepoch %3d : (loss)  %10f' % (
                        ep,
                        h_metrics['loss'][-1]))
                elif verbose != 0:
                    tqdm.write('epoch %3d : (loss)  %10f' % (
                        ep,
                        h_metrics['loss'][-1]))
    
    
        self.best_weights = cur_best_weights
        self.best_loss = cur_best_loss
    
        h_metrics['lrs'] = _lrs
        if not onecycle_mom_fun is None:
            h_metrics['moms'] = _moms
        h_metrics['loss_all'] = loss_all
        return h_metrics
