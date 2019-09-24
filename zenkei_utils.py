import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.colors as colors
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.patheffects as patheffects




def recover_img(img_norm, fixed_norm=False):
    """Recover properly scaling for image array from the scaled one.

    Parameters
    ----------
    img_norm : numpy array in float
        normalized image array.
    fixed_norm : bool
        Set True if the fixed scaling with
        128 mean and 80 std for uint8 data. Otherwise, standard
        dynamic normalization with the mean and the std.

    Returns
    -------
    (numpy array in float): the recovered image array whose range is [0, 1].
    """
    if fixed_norm:
        return np.clip((img_norm * 80.0 + 128.0) / 255.0, 0.0, 1.0)
    else:
        img_min = img_norm.min(axis=(0, 1))
        img_max = img_norm.max(axis=(0, 1))
        return (img_norm - img_min) / (img_max - img_min)


def get_imgs_hwc_from_chw(imgs_bchw):
    """Get images in (b, h, w, c) order from that in (b, c, h, w).

    Parameters
    ----------
    imgs_bchw (Torch Tensor or numpy array): images stored
        in (b, c, h, w) order.

    Returns
    -------
    (numpy array in float): images stored in (b, h, w, c) order.
    """
    is_torch = True if isinstance(imgs_bchw, torch.Tensor) else False

    shape = imgs_bchw.shape
    imgs = np.empty((0, shape[2], shape[3], shape[1]))
    for x in imgs_bchw:
        y = x.numpy() if is_torch else x

        # ch, h, w -> batch, h, w, ch
        # 0   1  2
        x_hwc = y.transpose((1, 2, 0))
        imgs = np.concatenate((imgs, np.expand_dims(x_hwc, 0)))
    return imgs




def plot_images(
    filepaths,
    sample_range=None,
    n_cols=4,
    labels=None,
    with_filename=False,
    use_draw_text=False,
    text_size=24,
    fsx=16,
    axis=None):
    """Plot images with list of file paths.

    Parameters
    ----------
    filepaths : list of str
        File paths of the image files.
    sample_range : list of int
        Indices of the images to plot in `filepaths`.
        If None is given, plot all images.
    n_cols : int
        Number of columns. Default is 4.
    labels : None or list of str
        The labels (titles) for each image.
        See also `use_draw_text` option.
    with_filename : bool
        If `labels` is `None`, then filename is shown as the title
        if it is `True`. Otherwise the index is shown.
    use_draw_text : bool
        If `labels` is given, the label is drawn on the picture
        if this is `True`. Otherwise, the label is on the title.
    text_size : int
        Font size of the text. Default is 24.
        This is effective only if `use_draw_text` is True.
    fsx : int
        Figure size factor for each element.
    axis : None or str
        Pass to `ax.axis()` if not `None` (default).
        Give `off` to hide the axis, for example.
    """

    if not isinstance(filepaths, (list, tuple)):
        raise ValueError('`filepaths` must be list or tuple')
    if sample_range == None:
        sample_range = list(range(len(filepaths)))
    if not labels is None:
        if not isinstance(labels, (list, tuple)):
            raise ValueError('`labels` must be list or tuple')
        if len(labels) != len(filepaths):
            raise ValueError('Lengths of `labels` and `filepaths` must be the same')

    n_rows = len(sample_range) // n_cols
    if n_cols * n_rows < len(sample_range): n_rows += 1

    #fsx = 16
    fsy = fsx / n_cols * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fsx, fsy))
    for i, ax in enumerate(axes.flat):
        if i >= len(sample_range):
            ax.axis('off')
            continue

        j = sample_range[i]
        img = Image.open(filepaths[j])
        if img.mode == 'CMYK': img = img.convert('RGB')
        ax.imshow(img)

        if use_draw_text and not labels is None:
            draw_text(ax, [0, 10], labels[j], sz=text_size)

        if not labels is None:
            if not use_draw_text:
                ax.set_title(labels[j])
        elif with_filename:
            ax.set_title(filepaths[j].split('/')[-1])
        else:
            ax.set_title(j)

        if not axis is None:
            ax.axis(axis)

    plt.tight_layout()


def plot_np_images(
    np_xs,
    sample_range=None,
    n_cols=4,
    labels=None,
    use_draw_text=False,
    text_size=24,
    fsx=16,
    is_torch=False,
    axis=None):
    """Plot images with list of file paths.

    Parameters
    ----------
    np_xs : numpy array
        Image data in the formats either numpy (b, h, w, c) or
        torch (b, c, h, w).
        See also `is_torch` option.
    sample_range : list of int
        Indices of the images to plot in `filepaths`.
        If None is given, plot all images.
    n_cols : int
        Number of columns. Default is 4.
    labels : None or list of str
        The labels (titles) for each image.
        If `None` is given, the index is used.
        See also `use_draw_text` option.
    use_draw_text : bool
        If `labels` is given, the label is drawn on the picture
        if this is `True`. Otherwise, the label is on the title.
    text_size : int
        Font size of the text. Default is 24.
        This is effective only if `use_draw_text` is True.
    fsx : int
        Figure size factor for each element.
    is_torch : bool
        Set `True` if `np_xs` is in torch format.
        Default is `False` for numpy format.
    axis : None or str
        Pass to `ax.axis()` if not `None` (default).
        Give `off` to hide the axis, for example.
    """

    if not isinstance(np_xs, np.ndarray): raise ValueError('np_xs is not numpy array')
    if len(np_xs.shape) != 4: raise ValueError('np_xs is wrong shape')
    if is_torch:
        # (batch, channel, height, width) to (batch, height, width, channel)
        np_xs_ = np_xs.transpose((0, 2, 3, 1))
    else:
        np_xs_ = np_xs
    xs_n, xs_h, xs_w, xs_c = np_xs_.shape
    if not xs_c in (1, 3): raise ValueError('np_xs is not GRAY or RGB image')

    if sample_range == None:
        sample_range = list(range(xs_n))

    if not labels is None:
        if not isinstance(labels, (list, tuple)):
            raise ValueError('`labels` must be list or tuple')
        if len(labels) != np_xs_.shape[0]:
            raise ValueError('Length of `labels` must be equal to the number of images given by `np_xs`.')

    n_rows = len(sample_range) // n_cols
    if n_cols * n_rows < len(sample_range): n_rows += 1

    #fsx = 16
    fsy = (fsx + 4) / n_cols * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fsx, fsy))
    for i, ax in enumerate(axes.flat):
        if i >= len(sample_range):
            ax.axis('off')
            continue

        j = sample_range[i]
        if xs_c == 1:
            # single channel image
            ax.imshow(recover_img(np_xs_[j])[:, : ,0])
        else:
            ax.imshow(recover_img(np_xs_[j]))

        if use_draw_text and not labels is None:
            draw_text(ax, [0, 10], labels[j], sz=text_size)

        if not labels is None:
            if not use_draw_text:
                ax.set_title(labels[j])
        else:
            ax.set_title(j)

        if not axis is None:
            ax.axis(axis)

    #plt.tight_layout()
    plt.show()




# from jeremy's lecture in 2018
def draw_outline(o, lw):
    o.set_path_effects([
        patheffects.Stroke(
            linewidth=lw,
            foreground='black'),
        patheffects.Normal()
    ])

def draw_text(ax, xy, txt, sz=20):
    text = ax.text(*xy, txt,
        verticalalignment='top',
        color='white',
        fontsize=sz,
        weight='bold')
    draw_outline(text, 2)


def plot_image_classification_results_with_preds(
    xs, ys, preds,
    sample_range,
    n_cols=4,
    labels=None,
    font_size=24,
    fsx=16,
    axis=None):
    """Plot the result of image classification with predictions.

    Parameters
    ----------
    xs : numpy.array
        Input data (images).
    ys : numpy array
        Labels (integers).
    preds : numpy array
        Predictions (integers).
    sample_range : list of int
        Indices of the images to plot in `xs`.
    n_cols : int
        Number of columns. Default is 4.
    labels : list of strings
    font_size : int
        Font size of the text. Default is 24.
        This is effective only if `use_draw_text` is True.
    fsx : int
        Figure size factor for each element.
    axis : None or str
        Pass to `ax.axis()` if not `None` (default).
        Give `off` to hide the axis, for example.
    """

    n_rows = len(sample_range) // n_cols
    if n_cols * n_rows < len(sample_range): n_rows += 1

    #fsx = 16
    fsy = fsx / n_cols * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fsx, fsy))
    for i, ax in enumerate(axes.flat):
        if i >= len(sample_range):
            ax.axis('off')
            continue

        j = sample_range[i]
        x = xs[j]
        true_it = ys[j]
        it = preds[j]

        #ax = plot_ax(x, ax=ax)

        # from (c, h, w) to (h, w, c)
        img = x.to(torch.device('cpu')).numpy().transpose((1, 2, 0))
        img = img - img.min(axis=(0, 1))
        img /= img.max(axis=(0, 1))

        if img.shape[2] == 1:
            # single channel image
            ax.imshow(img[:, :, 0])
        else:
            ax.imshow(img)

        if labels:
            cap = '予測：' + labels[it] + '\n正解：' + labels[true_it]
            draw_text(ax, [0, 10], cap, sz=font_size)

        if not axis is None:
            ax.axis(axis)

    #plt.tight_layout()
    plt.show()


# CAM plotting
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()


def plot_CAM_one(model, x, alpha=0.5, ax=None):
    if not ax: fig, ax = plt.subplots(figsize=(12, 8))

    # add cutoff in colormap
    cmap = plt.cm.jet
    #cmap = plt.cm.Purples

    # Get the colormap colors
    clrs_cmap = cmap(np.arange(cmap.N))
    #for i in range(cmap.N // 10):
    #    clrs_cmap[i, 3] = 0.0
    #cmap = colors.ListedColormap(clrs_cmap)
    new_clrs = np.empty(clrs_cmap.shape)
    for i in range(cmap.N // 2):
        new_clrs[i] = [0, 0, 0, 0]
        new_clrs[i + cmap.N // 2] = clrs_cmap[i*2]
    cmap = colors.ListedColormap(new_clrs)


    #sf = SaveFeatures(model.out)
    sf = SaveFeatures(model.final_bn)
    model.train(False)
    model.eval()
    py = model(torch.unsqueeze(x, 0))
    sf.remove()

    # handle output (probabilities)
    np_py = py.cpu().data.numpy()
    np_py = np.exp(np_py[0])
    img_type = np.argmax(np_py)

    # from (c, h, w) to (h, w, c)
    img = x.to(torch.device('cpu')).numpy().transpose((1, 2, 0))
    img = img - img.min(axis=(0, 1))
    img /= img.max(axis=(0, 1))

    if img.shape[2] == 1:
        # single channel image
        ax.imshow(img[:, :, 0])
    else:
        ax.imshow(img)

    # handle the features
    feat = np.maximum(0, sf.features[0])
    #print('feat.shape:', feat.shape)

    if feat.shape[-1] > 1:
        f2 = np.dot(np.rollaxis(feat, 0, 3), np_py)
        f2 -= f2.min()
        f2 /= f2.max()

        h, w, _ = img.shape
        f2_pil = Image.fromarray(np.uint8(f2*255.)).resize((w, h), Image.LANCZOS)
        #f2_pil = Image.fromarray(np.uint8(f2*255.)).resize((w, h), Image.NEAREST)
        f2_np = np.asarray(f2_pil)
        ax.imshow(f2_np, alpha=alpha, cmap=cmap)

    ax.axis('off')

    return ax, img_type


def plot_image_classification_results_with_CAM(
    model,
    xs, ys,
    sample_range,
    n_cols=4,
    alpha=0.5,
    labels=None,
    font_size=24,
    fsx=16,
    axis=None):
    """Plot the result of image classification with class activation map (CAM).

    Parameters
    ----------
    model : nn.Module
        The model to predict used to get CAM (and the predictions as well).
    xs : numpy.array
        Input data (images).
    ys : numpy array
        Labels (int).
    sample_range : list of int
        Indices of the images to plot in `xs`.
    n_cols : int
        Number of columns. Default is 4.
    alpha : float
        Transparency of CAM layer. Default is 0.5.
    labels : list of strings
        Table from class index to class string.
    font_size : int
        Font size of the text. Default is 24.
        This is effective only if `use_draw_text` is True.
    fsx : int
        Figure size factor for each element.
    axis : None or str
        Pass to `ax.axis()` if not `None` (default).
        Give `off` to hide the axis, for example.
    """

    n_rows = len(sample_range) // n_cols
    if n_cols * n_rows < len(sample_range): n_rows += 1

    #fsx = 16
    fsy = fsx / n_cols * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fsx, fsy))
    for i, ax in enumerate(axes.flat):
        if i >= len(sample_range):
            ax.axis('off')
            continue

        j = sample_range[i]
        x = xs[j]
        true_it = ys[j]

        ax, it = plot_CAM_one(model, x, alpha=alpha, ax=ax)
        if labels:
            cap = '予測：' + labels[it] + '\n正解：' + labels[true_it]
            draw_text(ax, [0, 10], cap, sz=font_size)

        if not axis is None:
            ax.axis(axis)

    #plt.tight_layout()
    plt.show()
