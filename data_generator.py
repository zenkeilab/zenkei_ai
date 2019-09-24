import numpy as np
import torch

import random
import math
from PIL import Image

import gzip
import struct


def read_mnist(filename):
    """ Read MNIST dataset such as `idx3-ubyte` and `idx1-ubyte`.

    Parameters
    ----------
    filename : str
        File path of the MNIST file. For example,
    `../some_place/train-images-idx3-ubyte` or
    `../other_place/t10k-labels-idx1-ubyte.gz`.
    """
    if filename.split('.')[-1] == 'gz':
        f = gzip.open(filename, 'rb')
    else:
        f = open(filename, mode='rb')

    # read header
    d = f.read(4)
    if d == b'\x00\x00\x08\03':
        is_label = False
    elif d == b'\x00\x00\x08\01':
        is_label = True

    if is_label:
        n = struct.unpack('>I', f.read(4))[0]
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    else:
        n = struct.unpack('>I', f.read(4))[0]
        h = struct.unpack('>I', f.read(4))[0]
        w = struct.unpack('>I', f.read(4))[0]
        data = np.frombuffer(f.read(n*h*w), dtype=np.uint8)
        data = np.reshape(data, (n, h, w))

    f.close()
    return data




def norm_angle(theta):
    # normalize theta in [-pi, pi]
    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    return math.atan2(stheta, ctheta)

# pano : panorama image
# dyaw : rotation angle of camera in radian
#        right is positive
def rotate_pano(pano, dyaw):

    if len(pano.shape) != 3: return None
    h, w, ch = pano.shape

    # normalize dyaw in [-pi, pi]
    dyaw_norm = norm_angle(dyaw)

    # shift pixels
    dw = dyaw_norm / (2.0 * math.pi) * float(w)

    if dw > 0.0:
        iw = int(dw+0.5)
        pano_l = pano[:, :iw, :]
        pano_r = pano[:, iw:, :]
        return np.concatenate((pano_r, pano_l), axis=1)
    elif dw < 0.0:
        iw = w + int(dw+0.5)
        pano_l = pano[:, :iw, :]
        pano_r = pano[:, iw:, :]
        return np.concatenate((pano_r, pano_l), axis=1)
    else: return pano


def image_ch_swap(x):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[i,j] = (x[i,j,2], x[i,j,1], x[i,j,0])
    return y




def distort_image_with_params(
    np_image,
    image_size=224,
    keep_aspect_ratio=True,
    flip_horizontal=False,
    flip_vertical=False,
    dist_pano_dyaw=0,
    dist_zoom_rands=None,
    dist_rot_angle=0,
    dist_rot_n_angle=0,
    dist_shift_x_px=0,
    dist_shift_y_px=0,
    fill_ave=False):
    """ Distort image.

    Parameters
    ----------
    np_image : numpy.array
        The original image.
    image_size : int or list of ints
        The size of output image in xs.
    keep_aspect_ratio : bool
        If set True, the aspect ratio is kept.
        Otherwise, the image is just resized to 'image_size x image_size'.
    flip_horizontal : bool
        Set True then image is flipped horizontally in 50% probability.
    flip_vertical : bool
        Set True then image is flipped vertically in 50% probability.
    dist_zoom : bool
        If set True, the original image is scaled by random factor in the range of
        (1.0, dist_zoom_max), where dist_zoom_max is given by the option 'dist_zoom_max'.
    dist_zoom_max : float
        The maximum zoom ratio (effective only if dist_zoom is True).
        The default value is 1.2.
    dist_rot : bool
        If set True, the original image is rotated by random angle in the range of
        (-dist_rot_max, dist_rot_max), where dist_rot_max is given by the option 'dist_rot_max'.
    dist_rot_max : float
        The maximum angle to rotation distortion (effective only if dist_rot is True).
        The default value is 10. The unit is degree.
    dist_rot_n : int
        Discrete rotation distortion. Default value (0) means no distortion.
        The number `n` represents the division of 360 degree.
        Therefore `n=1` also means no distortion.
        For `n=2`, the image is rotated 180 degree in 50% probability.
        For `n=3`, rotated 0, 120, or 240 degrees in 33% each. And so on.
    dist_shift_x : int
        Distortion of horizontal shift by pixel.
        Default value (0) means no distortion.
        Give positive value `x`, then the actual shift is chosen randomly
        in the range of `[-x, x]`.
    dist_shift_y : int
        Distortion of vertical shift by pixel.
        Default value (0) means no distortion.
        Give positive value `y`, then the actual shift is chosen randomly
        in the range of `[-y, y]`.
    fill_ave : bool
        If set `True`, the black pixel `(0, 0, 0)` in `uint8` format
        will be replaced by the average color of the original image.
        Note that the black pixels existing on the original are also
        replaced as well as those introduced by rotation and shifting.
    """

    if not isinstance(np_image, np.ndarray):
        raise ValueError('`np_image` must be numpy array')

    if isinstance(image_size, list) or isinstance(image_size, tuple):
        if len(image_size) != 2:
            raise ValueError('if `image_size` is list, the length must be 2')
        image_width = image_size[0]
        image_height = image_size[1]
    elif isinstance(image_size, int):
        image_width = image_size
        image_height = image_size
    else:
        raise ValueError('`image_size` must be either value or list of values')

    if not dist_zoom_rands is None:
        if not isinstance(dist_zoom_rands, (list, tuple)):
            raise ValueError('`dist_zoom_rands` must be None or list or tuple')
        if len(dist_zoom_rands) != 4:
            raise ValueError('length of `dist_zoom_rands` must be 4')

    # check gray-scale image or not (like MNIST dataset)
    is_gray_image = True if len(np_image.shape) != 3 else False

    av_color = None
    if fill_ave:
        av_color = np_image.mean(axis=(0, 1))


    # convert image from np to PIL
    img = Image.fromarray(np_image)

    # set scale
    w_orig, h_orig = img.size
    scale_w = float(image_width) / float(w_orig)
    scale_h = float(image_height) / float(h_orig)
    if keep_aspect_ratio:
        # keep aspect ratio
        if scale_w > scale_h:
            scale_h = scale_w
        else:
            scale_w = scale_h

    if not dist_zoom_rands is None:
        if keep_aspect_ratio:
            scale_w *= dist_zoom_rands[0]
            scale_h *= dist_zoom_rands[0]
        else:
            scale_w *= dist_zoom_rands[0]
            scale_h *= dist_zoom_rands[1]

    # apply rotation on PIL
    if dist_rot_angle != 0.:
        img = img.rotate(dist_rot_angle, expand=True)
    if dist_rot_n_angle != 0.:
        img = img.rotate(dist_rot_n_angle, expand=True)

    # shift
    sx = dist_shift_x_px / scale_w
    sy = dist_shift_y_px / scale_h
    theta = (dist_rot_angle + dist_rot_n_angle) * math.pi / 180.0
    ct = math.cos(-theta)
    st = math.sin(-theta)
    shift_x = ct * sx + st * sy
    shift_y = -(-st * sx + ct * sy)

    # crop on the original scale
    w, h = img.size
    w_crop = float(image_width) / scale_w
    h_crop = float(image_height) / scale_h

    x0_crop = 0.5 * (w - w_crop) + shift_x
    y0_crop = 0.5 * (h - h_crop) + shift_y
    img_crop = img.crop((x0_crop, y0_crop, x0_crop+w_crop, y0_crop+h_crop))

    # resize by PIL
    img_scaled = img_crop.resize((image_width, image_height))
    # NOW I'M HERE!!

    # convert to numpy
    np_img = np.array(img_scaled)

    if fill_ave:
        np_img[np.where(
            (np_img[:, :, 0] == 0) &
            (np_img[:, :, 1] == 0) &
            (np_img[:, :, 2] == 0)
        )] = av_color

    # normalize from [0, 255] in uint8 to [0, 1] in float32
    img = (np_img / 255).astype(np.float32)

    # flip
    if is_gray_image:
        if flip_horizontal: img = img[:, ::-1]
        if flip_vertical: img = img[::-1, :]
    else:
        if flip_horizontal: img = img[:, ::-1, :]
        if flip_vertical: img = img[::-1, :, :]

    # pano distortion
    if dist_pano_dyaw != 0.:
        img = rotate_pano(img, dist_pano_dyaw)

    return img

def distort_image(
    np_image,
    image_size=224,
    keep_aspect_ratio=True,
    flip_horizontal=False, flip_vertical=False,
    dist_pano=False,
    dist_zoom=False, dist_zoom_min=1.0, dist_zoom_max=1.2,
    dist_rot=False, dist_rot_max=10,
    dist_rot_n=0,
    dist_shift_x=0, dist_shift_y=0,
    fill_ave=False):
    """ Distort image.

    Parameters
    ----------
    np_image : numpy.array
        The original image.
    image_size : int or list of ints
        The size of output image in xs.
    keep_aspect_ratio : bool
        If set True, the aspect ratio is kept.
        Otherwise, the image is just resized to 'image_size x image_size'.
    flip_horizontal : bool
        Set True then image is flipped horizontally in 50% probability.
    flip_vertical : bool
        Set True then image is flipped vertically in 50% probability.
    dist_zoom : bool
        If set `True`, the original image is scaled by a random factor
        in the range of `(dist_zoom_min, dist_zoom_max)`, where
        `dist_zoom_min` and `dist_zoom_max` are given by the options
        'dist_zoom_min' and 'dist_zoom_max'.
    dist_zoom_min : float
        The minimum zoom ratio (effective only if dist_zoom is True).
        The default value is 1.0.
    dist_zoom_max : float
        The maximum zoom ratio (effective only if dist_zoom is True).
        The default value is 1.2.
    dist_rot : bool
        If set True, the original image is rotated by random angle in the range of
        (-dist_rot_max, dist_rot_max), where dist_rot_max is given by the option 'dist_rot_max'.
    dist_rot_max : float
        The maximum angle to rotation distortion (effective only if dist_rot is True).
        The default value is 10. The unit is degree.
    dist_rot_n : int
        Discrete rotation distortion. Default value (0) means no distortion.
        The number `n` represents the division of 360 degree.
        Therefore `n=1` also means no distortion.
        For `n=2`, the image is rotated 180 degree in 50% probability.
        For `n=3`, rotated 0, 120, or 240 degrees in 33% each. And so on.
    dist_shift_x : int
        Distortion of horizontal shift by pixel.
        Default value (0) means no distortion.
        Give positive value `x`, then the actual shift is chosen randomly
        in the range of `[-x, x]`.
    dist_shift_y : int
        Distortion of vertical shift by pixel.
        Default value (0) means no distortion.
        Give positive value `y`, then the actual shift is chosen randomly
        in the range of `[-y, y]`.
    fill_ave : bool
        If set `True`, the black pixel `(0, 0, 0)` in `uint8` format
        will be replaced by the average color of the original image.
        Note that the black pixels existing on the original are also
        replaced as well as those introduced by rotation and shifting.
    """

    if not isinstance(np_image, np.ndarray):
        raise ValueError('`np_image` must be numpy array')

    if isinstance(image_size, list) or isinstance(image_size, tuple):
        if len(image_size) != 2:
            raise ValueError('if `image_size` is list, the length must be 2')
    elif not isinstance(image_size, int):
        raise ValueError('`image_size` must be either value or list of values')


    dist_rot_angle = random.uniform(-dist_rot_max, dist_rot_max) if dist_rot else 0

    dist_rot_n_angle = 360.0 / dist_rot_n * random.randrange(dist_rot_n) if dist_rot_n != 0 else 0

    dist_zoom_rands = [
        random.uniform(dist_zoom_min, dist_zoom_max),
        random.uniform(dist_zoom_min, dist_zoom_max),
        random.uniform(0, 1),
        random.uniform(0, 1)
    ] if dist_zoom else None

    flip_horizontal = flip_horizontal and random.random() < 0.5
    flip_vertical   = flip_vertical and random.random() < 0.5

    dist_pano_dyaw = random.random() * math.pi * 2.0 if dist_pano else 0

    dist_shift_x_px = random.randrange(
        -dist_shift_x, dist_shift_x+1) if dist_shift_x else 0
    dist_shift_y_px = random.randrange(
        -dist_shift_y, dist_shift_y+1) if dist_shift_y else 0


    return distort_image_with_params(
        np_image,
        image_size=image_size,
        keep_aspect_ratio=keep_aspect_ratio,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
        dist_pano_dyaw=dist_pano_dyaw,
        dist_zoom_rands=dist_zoom_rands,
        dist_rot_angle=dist_rot_angle,
        dist_rot_n_angle=dist_rot_n_angle,
        dist_shift_x_px=dist_shift_x_px,
        dist_shift_y_px=dist_shift_y_px,
        fill_ave=fill_ave)


def data_generator(
    batch_size,
    file_list,
    label_list,
    shuffle=False,
    is_gray_image=False,
    bgr=False,
    fixed_norm=None,
    no_std_norm=False,
    min_max_norm=False,
    output_format='numpy',
    **kwargs):
    """Data generator.

    Parameters
    ----------
    batch_size : int
        The size of a batch.
    file_list :
        A list of filenames of images.
    label_list :
        A list of the label corresponding to file_list.
        if None is given, return only xs (for prediction).
    shuffle : bool
        Set True if you want to shuffle the sequence of data. Default value is True.
    is_gray_image : bool
        The image data is formed as 1 channel array.
        Default is False.
    bgr : bool
        Set True if the order of image channel should be BGR.
        If `is_gray_image == False`, just ignored.
        Default value is Fasle where the order is RGB.
    bgr : bool
        Set True if the order of image channel should be BGR.
        Default value is Fasle where the order is RGB.
    fixed_norm : None or [np.array, nparray]
        If `None`, normalize the pixel value for each image.
        Otherwise, use the first element `fixed_norm[0]` for the mean,
        the second element `fixed_norm[1]` for the standard deviation.
    no_std_norm : bool
        If `fixed_norm` is not `None`, this option has no effect. Otherwise,
        if set True, pixel values are not divided by their standard deviation, so that
        the range of the value would be like [-128, 128], while if set False, divided by the standard
        deviation and the range would be like [-1, 1].
    min_max_norm : bool
        If set `True` for `min_max_norm`, the options `fixed_norm` and `no_std_norm` are just ignored.
        The pixel values are normalized to [0, 1] by the minimum and the maximum.
    output_format : string
        Either 'numpy' or 'torch'.

    **kwargs :
        for distort_image()
    image_size : int or list of ints
        The size of output image in xs.
    keep_aspect_ratio : bool
        If set True, the aspect ratio is kept.
        Otherwise, the image is just resized to 'image_size x image_size'.
    flip_horizontal : bool
        Set True then image is flipped horizontally in 50% probability.
    flip_vertical : bool
        Set True then image is flipped vertically in 50% probability.
    dist_zoom : bool
        If set `True`, the original image is scaled by a random factor
        in the range of `(dist_zoom_min, dist_zoom_max)`, where
        `dist_zoom_min` and `dist_zoom_max` are given by the options
        'dist_zoom_min' and 'dist_zoom_max'.
    dist_zoom_min : float
        The minimum zoom ratio (effective only if dist_zoom is True).
        The default value is 1.0.
    dist_zoom_max : float
        The maximum zoom ratio (effective only if dist_zoom is True).
        The default value is 1.2.
    dist_rot : bool
        If set True, the original image is rotated by random angle in the range of
        (-dist_rot_max, dist_rot_max), where dist_rot_max is given by the option 'dist_rot_max'.
    dist_rot_max : float
        The maximum angle to rotation distortion (effective only if dist_rot is True).
        The default value is 10. The unit is degree.
    dist_rot_n : int
        Discrete rotation distortion. Default value (0) means no distortion.
        The number `n` represents the division of 360 degree.
        Therefore `n=1` also means no distortion.
        For `n=2`, the image is rotated 180 degree in 50% probability.
        For `n=3`, rotated 0, 120, or 240 degrees in 33% each. And so on.
    dist_shift_x : int
        Distortion of horizontal shift by pixel.
        Default value (0) means no distortion.
        Give positive value `x`, then the actual shift is chosen randomly
        in the range of `[-x, x]`.
        Blank pixels are filled by zero after normalization applied.
    dist_shift_y : int
        Distortion of vertical shift by pixel.
        Default value (0) means no distortion.
        Give positive value `y`, then the actual shift is chosen randomly
        in the range of `[-y, y]`.
        Blank pixels are filled by zero after normalization applied.
    fill_ave : bool
        If set `True`, the black pixel `(0, 0, 0)` in `uint8` format
        will be replaced by the average color of the original image.
        Note that the black pixels existing on the original are also
        replaced as well as those introduced by rotation and shifting.


    Returns
    -------
        Two numpy arrays (xs, ys) are returned if label_list is given, where xs is the image data in the shape of
        (batch_size, 224, 224, 3) and ys is the corredponding labels in the shape of (batch_size, ).
        If label_list is None, only xs is returned.
    """
    if output_format != 'numpy' and output_format != 'torch':
        raise ValueError("output_format: invalid value. it should be either 'numpy' or 'torch'.")


    # check the label is in sparse or in one-hot (or multi-label) format
    if not label_list is None:
        label_list_shape = np.array(label_list).shape
        if len(label_list_shape) < 1 or len(label_list_shape) > 2:
            raise ValueError('label_list is wrong format')

        if len(file_list) != label_list_shape[0]:
            raise ValueError('file_list and label_list mismatch')
        is_sparse_label = True if len(label_list_shape) == 1 else False

    if not fixed_norm is None:
        if not isinstance(fixed_norm, (list, tuple)):
            raise ValueError('fixed_norm must be a list')
        if len(fixed_norm) != 2:
            raise ValueError('fixed_norm must be a list of two elements')
        if not (isinstance(fixed_norm[0], np.ndarray) and
                isinstance(fixed_norm[1], np.ndarray)):
            raise ValueError('the elements of fixed_norm must be np.array')


    n_batches = len(file_list) // batch_size


    idx = list(range(len(file_list)))
    while 1:
        if shuffle: random.shuffle(idx)
        for i in range(n_batches):
            xs = None
            ys = []
            for j in range(batch_size):
                k = idx[i * batch_size + j]

                filename = file_list[k]
                img_orig = Image.open(filename)
                # it seems that some image is greyscale
                if is_gray_image:
                    if img_orig.mode != 'L': img_orig = img_orig.convert('L')
                else:
                    if img_orig.mode != 'RGB': img_orig = img_orig.convert('RGB')

                img = distort_image(
                    np.array(img_orig),
                    **kwargs)

                if min_max_norm:
                    img = img / (img.max(axis=(0, 1)) + 1.0e-6)
                elif fixed_norm is None:
                    img = (img - np.mean(img, axis=(0,1)))
                    if not no_std_norm: img = img / (np.std(img, axis=(0,1)) + 1.0e-6)
                else:
                    img = (img - fixed_norm[0])
                    img = img / (fixed_norm[1] + 1.0e-6)

                if not is_gray_image and bgr: img = image_ch_swap(img)

                if xs is None: xs = np.expand_dims(img, axis=0)
                else: xs = np.concatenate((xs, np.expand_dims(img, axis=0)))

                if not label_list is None: ys.append(label_list[k])

            # return one batch data
            if output_format == 'torch':
                # for pytorch

                # from (batch, height, width, channel) to (batch, channel, height, width)
                #       0      1       2      3
                if is_gray_image:
                    xs_ = np.expand_dims(xs, 1)
                else:
                    xs_ = xs.transpose((0, 3, 1, 2))

                if label_list is None:
                    yield torch.from_numpy(xs_).float()
                elif is_sparse_label:
                    ys_ = np.array(ys)
                    yield torch.from_numpy(xs_).float(), torch.LongTensor(ys_.reshape((-1)))
                else:
                    ys_ = np.array(ys)
                    yield torch.from_numpy(xs_).float(), torch.from_numpy(ys_).float()

            else:
                # for keras

                if label_list is None:
                    yield xs
                else:
                    yield xs, np.array(ys)


def np_data_generator(
    batch_size,
    np_xs,
    np_ys,
    shuffle=False,
    bgr=False,
    fixed_norm=None,
    no_std_norm=False,
    min_max_norm=False,
    output_format='numpy',
    **kwargs):
    """Data generator from numpy arrays.

    Parameters
    ----------
    batch_size : int
        The size of a batch.
    np_xs : numpy array
        Image data
    np_ys : numpy array
        Labels corresponding to `np_xs`.
    shuffle : bool
        Set True if you want to shuffle the sequence of data. Default value is True.
    bgr : bool
        Set True if the order of image channel should be BGR.
        Default value is Fasle where the order is RGB.
    fixed_norm : None or [np.array, nparray]
        If `None`, normalize the pixel value for each image.
        Otherwise, use the first element `fixed_norm[0]` for the mean,
        the second element `fixed_norm[1]` for the standard deviation.
    no_std_norm : bool
        If `fixed_norm` is not `None`, this option has no effect. Otherwise,
        if set True, pixel values are not divided by their standard deviation, so that
        the range of the value would be like [-128, 128], while if set False, divided by the standard
        deviation and the range would be like [-1, 1].
    min_max_norm : bool
        If set `True` for `min_max_norm`, the options `fixed_norm` and `no_std_norm` are just ignored.
        The pixel values are normalized to [0, 1] by the minimum and the maximum.
    output_format : string
        Either 'numpy' or 'torch'.

    **kwargs :
        for distort_image()
    image_size : int or list of ints
        The size of output image in xs.
    keep_aspect_ratio : bool
        If set True, the aspect ratio is kept.
        Otherwise, the image is just resized to 'image_size x image_size'.
    flip_horizontal : bool
        Set True then image is flipped horizontally in 50% probability.
    flip_vertical : bool
        Set True then image is flipped vertically in 50% probability.
    dist_zoom : bool
        If set `True`, the original image is scaled by a random factor
        in the range of `(dist_zoom_min, dist_zoom_max)`, where
        `dist_zoom_min` and `dist_zoom_max` are given by the options
        'dist_zoom_min' and 'dist_zoom_max'.
    dist_zoom_min : float
        The minimum zoom ratio (effective only if dist_zoom is True).
        The default value is 1.0.
    dist_zoom_max : float
        The maximum zoom ratio (effective only if dist_zoom is True).
        The default value is 1.2.
    dist_rot : bool
        If set True, the original image is rotated by random angle in the range of
        (-dist_rot_max, dist_rot_max), where dist_rot_max is given by the option 'dist_rot_max'.
    dist_rot_max : float
        The maximum angle to rotation distortion (effective only if dist_rot is True).
        The default value is 10. The unit is degree.
    dist_rot_n : int
        Discrete rotation distortion. Default value (0) means no distortion.
        The number `n` represents the division of 360 degree.
        Therefore `n=1` also means no distortion.
        For `n=2`, the image is rotated 180 degree in 50% probability.
        For `n=3`, rotated 0, 120, or 240 degrees in 33% each. And so on.
    dist_shift_x : int
        Distortion of horizontal shift by pixel.
        Default value (0) means no distortion.
        Give positive value `x`, then the actual shift is chosen randomly
        in the range of `[-x, x]`.
    dist_shift_y : int
        Distortion of vertical shift by pixel.
        Default value (0) means no distortion.
        Give positive value `y`, then the actual shift is chosen randomly
        in the range of `[-y, y]`.
    fill_ave : bool
        If set `True`, the black pixel `(0, 0, 0)` in `uint8` format
        will be replaced by the average color of the original image.
        Note that the black pixels existing on the original are also
        replaced as well as those introduced by rotation and shifting.


    Returns
    -------
        Two numpy arrays (xs, ys) are returned if label_list is given, where xs is the image data in the shape of
        (batch_size, 224, 224, 3) and ys is the corredponding labels in the shape of (batch_size, ).
        If label_list is None, only xs is returned.
    """
    if output_format != 'numpy' and output_format != 'torch':
        raise ValueError("output_format: invalid value. it should be either 'numpy' or 'torch'.")


    # check the label is in sparse or in one-hot (or multi-label) format
    if not isinstance(np_xs, np.ndarray): raise ValueError('np_xs is not numpy array')
    if not isinstance(np_ys, np.ndarray): raise ValueError('np_ys is not numpy array')

    if len(np_xs.shape) == 4:
        is_gray_image = False
        xs_n, xs_h, xs_w, xs_c = np_xs.shape
        #if xs_c != 3: raise ValueError('np_xs is not RGB image')
    elif len(np_xs.shape) == 3:
        is_gray_image = True
        xs_n, xs_h, xs_w = np_xs.shape
    else:
        raise ValueError('np_xs is wrong shape')

    if not len(np_ys.shape) in (1, 2): raise ValueError('np_ys is wrong shape')
    if len(np_ys.shape) == 1:
        ys_n, = np_ys.shape
        is_sparse_label = True
    else:
        ys_n, ys_nclass = np_ys.shape
        is_sparse_label = False

    if xs_n != ys_n: raise ValueError('sample sizes mismatch')


    if not fixed_norm is None:
        if not isinstance(fixed_norm, (list, tuple)):
            raise ValueError('fixed_norm must be a list')
        if len(fixed_norm) != 2:
            raise ValueError('fixed_norm must be a list of two elements')
        if not (isinstance(fixed_norm[0], np.ndarray) and
                isinstance(fixed_norm[1], np.ndarray)):
            raise ValueError('the elements of fixed_norm must be np.array')


    n_batches = xs_n // batch_size


    idx = list(range(xs_n))
    while 1:
        if shuffle: random.shuffle(idx)
        for i in range(n_batches):
            xs = None
            ys = []
            for j in range(batch_size):
                k = idx[i * batch_size + j]

                if is_gray_image:
                    img = distort_image(
                        np_xs[k, :, :],
                        **kwargs)
                else:
                    img = distort_image(
                        np_xs[k, :, :, :],
                        **kwargs)

                if min_max_norm:
                    img = img / (img.max(axis=(0, 1)) + 1.0e-6)
                elif fixed_norm is None:
                    img = (img - np.mean(img, axis=(0,1)))
                    if not no_std_norm: img = img / (np.std(img, axis=(0,1)) + 1.0e-6)
                else:
                    img = (img - fixed_norm[0])
                    img = img / (fixed_norm[1] + 1.0e-6)

                if bgr: img = image_ch_swap(img)

                if xs is None: xs = np.expand_dims(img, axis=0)
                else: xs = np.concatenate((xs, np.expand_dims(img, axis=0)))

                #if not label_list is None: ys.append(label_list[k])
                ys.append(np_ys[k])

            # return one batch data
            if output_format == 'torch':
                # for pytorch

                # from (batch, height, width, channel) to (batch, channel, height, width)
                #       0      1       2      3
                if is_gray_image:
                    xs_ = np.expand_dims(xs, 1)
                else:
                    xs_ = xs.transpose((0, 3, 1, 2))

                if is_sparse_label:
                    ys_ = np.array(ys)
                    yield torch.from_numpy(xs_).float(), torch.LongTensor(ys_.reshape((-1)))
                else:
                    ys_ = np.array(ys)
                    yield torch.from_numpy(xs_).float(), torch.from_numpy(ys_).float()

            else:
                # for keras
                yield xs, np.array(ys)


def img_to_img_data_generator(
    batch_size,
    input_list,
    output_list,
    shuffle=False,
    image_size=None,
    keep_aspect_ratio=True,
    flip_horizontal=False, flip_vertical=False,
    dist_pano=False,
    dist_zoom=False, dist_zoom_min=1.0, dist_zoom_max=1.2,
    dist_rot=False, dist_rot_max=10,
    dist_rot_n=0,
    dist_shift_x=0, dist_shift_y=0,
    fill_ave=False,
    bgr=False,
    fixed_norm=None,
    no_std_norm=False,
    output_format='numpy'):
    """Data generator for image input and image output.

    Parameters
    ----------
    batch_size : int
        The size of a batch.
    input_list:
        A list of filenames of input images.
    output_list:
        A list of filenames of output images.
    shuffle : bool
        Set True if you want to shuffle the sequence of data. Default value is True.
    image_size : None or int
        If `None` is given, the original size is used.
    keep_aspect_ratio : bool
        If set True, the aspect ratio is kept.
        Otherwise, the image is just resized to 'image_size x image_size'.
    flip_horizontal : bool
        Set True then image is flipped horizontally in 50% probability.
    flip_vertical : bool
        Set True then image is flipped vertically in 50% probability.
    dist_zoom : bool
        If set `True`, the original image is scaled by a random factor
        in the range of `(dist_zoom_min, dist_zoom_max)`, where
        `dist_zoom_min` and `dist_zoom_max` are given by the options
        'dist_zoom_min' and 'dist_zoom_max'.
    dist_zoom_min : float
        The minimum zoom ratio (effective only if dist_zoom is True).
        The default value is 1.0.
    dist_zoom_max : float
        The maximum zoom ratio (effective only if dist_zoom is True).
        The default value is 1.2.
    dist_rot : bool
        If set True, the original image is rotated by random angle in the range of
        (-dist_rot_max, dist_rot_max), where dist_rot_max is given by the option 'dist_rot_max'.
    dist_rot_max : float
        The maximum angle to rotation distortion (effective only if dist_rot is True).
        The default value is 10. The unit is degree.
    dist_rot_n : int
        Discrete rotation distortion. Default value (0) means no distortion.
        The number `n` represents the division of 360 degree.
        Therefore `n=1` also means no distortion.
        For `n=2`, the image is rotated 180 degree in 50% probability.
        For `n=3`, rotated 0, 120, or 240 degrees in 33% each. And so on.
    dist_shift_x : int
        Distortion of horizontal shift by pixel.
        Default value (0) means no distortion.
        Give positive value `x`, then the actual shift is chosen randomly
        in the range of `[-x, x]`.
    dist_shift_y : int
        Distortion of vertical shift by pixel.
        Default value (0) means no distortion.
        Give positive value `y`, then the actual shift is chosen randomly
        in the range of `[-y, y]`.
    fill_ave : bool
        If set `True`, the black pixel `(0, 0, 0)` in `uint8` format
        will be replaced by the average color of the original image.
        Note that the black pixels existing on the original are also
        replaced as well as those introduced by rotation and shifting.
    bgr : bool
        Set True if the order of image channel should be BGR.
        Default value is Fasle where the order is RGB.
    fixed_norm : None or [np.array, nparray]
        If `None`, normalize the pixel value for each image.
        Otherwise, use the first element `fixed_norm[0]` for the mean,
        the second element `fixed_norm[1]` for the standard deviation.
    no_std_norm : bool
        If `fixed_norm` is not `None`, this option has no effect. Otherwise,
        if set True, pixel values are not divided by their standard deviation, so that
        the range of the value would be like [-128, 128], while if set False, divided by the standard
        deviation and the range would be like [-1, 1].
    output_format : string
        Either 'numpy' or 'torch'.

    Returns:
    -------
        Two numpy arrays (xs, ys) are returned if label_list is given, where xs is the image data in the shape of
        (batch_size, 224, 224, 3) and ys is the corredponding labels in the shape of (batch_size, ).
        If label_list is None, only xs is returned.
    """
    if output_format != 'numpy' and output_format != 'torch':
        raise ValueError("output_format: invalid value. it should be either 'numpy' or 'torch'.")


    if len(input_list) != len(output_list):
        raise ValueError('input_list and output_list mismatch')


    n_batches = len(input_list) // batch_size


    idx = list(range(len(input_list)))
    while 1:
        if shuffle: random.shuffle(idx)
        for i in range(n_batches):
            xs = None
            ys = None
            for j in range(batch_size):
                k = idx[i * batch_size + j]

                input_file = input_list[k]
                output_file = output_list[k]
                input_img = Image.open(input_file)
                output_img = Image.open(output_file)

                # it seems that some image is greyscale
                if input_img.mode != 'RGB': input_img = input_img.convert('RGB')
                if output_img.mode != 'RGB': output_img = output_img.convert('RGB')

                # set random parameters
                dist_rot_angle = random.uniform(
                    -dist_rot_max, dist_rot_max) if dist_rot else 0

                dist_rot_n_angle = 360.0 / dist_rot_n * random.randrange(
                    dist_rot_n) if dist_rot_n != 0 else 0

                dist_zoom_rands = [
                    random.uniform(dist_zoom_min, dist_zoom_max),
                    random.uniform(dist_zoom_min, dist_zoom_max),
                    random.uniform(0, 1),
                    random.uniform(0, 1)
                ] if dist_zoom else None

                flip_horizontal = flip_horizontal and random.random() < 0.5
                flip_vertical   = flip_vertical and random.random() < 0.5

                dist_pano_dyaw = random.random() * math.pi * 2.0 if dist_pano else 0

                dist_shift_x_px = random.randrange(
                    -dist_shift_x, dist_shift_x+1) if dist_shift_x else 0
                dist_shift_y_px = random.randrange(
                    -dist_shift_y, dist_shift_y+1) if dist_shift_y else 0

                # apply the same distortion for both input and output
                if image_size is None: im_sz = input_img.size
                else: im_sz = image_size
                img_in = distort_image_with_params(
                    np.array(input_img),
                    image_size=im_sz,
                    keep_aspect_ratio=keep_aspect_ratio,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    dist_pano_dyaw=dist_pano_dyaw,
                    dist_zoom_rands=dist_zoom_rands,
                    dist_rot_angle=dist_rot_angle,
                    dist_rot_n_angle=dist_rot_n_angle,
                    dist_shift_x_px=dist_shift_x_px,
                    dist_shift_y_px=dist_shift_y_px,
                    fill_ave=fill_ave)
                if image_size is None: im_sz = output_img.size
                else: im_sz = image_size
                img_out = distort_image_with_params(
                    np.array(output_img),
                    image_size=im_sz,
                    keep_aspect_ratio=keep_aspect_ratio,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    dist_pano_dyaw=dist_pano_dyaw,
                    dist_zoom_rands=dist_zoom_rands,
                    dist_rot_angle=dist_rot_angle,
                    dist_rot_n_angle=dist_rot_n_angle,
                    dist_shift_x_px=dist_shift_x_px,
                    dist_shift_y_px=dist_shift_y_px,
                    fill_ave=fill_ave)


                if fixed_norm is None:
                    img_in = (img_in - np.mean(img_in, axis=(0,1)))
                    if not no_std_norm: img_in = img_in / (np.std(img_in, axis=(0,1)) + 1.0e-6)
                    img_out = (img_out - np.mean(img_out, axis=(0,1)))
                    if not no_std_norm: img_out = img_out / (np.std(img_out, axis=(0,1)) + 1.0e-6)
                else:
                    img_in = (img_in - fixed_norm[0])
                    img_in = img_in / (fixed_norm[1] + 1.0e-6)
                    img_out = (img_out - fixed_norm[0])
                    img_out = img_out / (fixed_norm[1] + 1.0e-6)

                if bgr:
                    img_in = image_ch_swap(img_in)
                    img_out = image_ch_swap(img_out)

                if xs is None: xs = np.expand_dims(img_in, axis=0)
                else: xs = np.concatenate((xs, np.expand_dims(img_in, axis=0)))

                if ys is None: ys = np.expand_dims(img_out, axis=0)
                else: ys = np.concatenate((ys, np.expand_dims(img_out, axis=0)))


            # return one batch data
            if output_format == 'torch':
                # for pytorch

                # from (batch, height, width, channel) to (batch, channel, height, width)
                #       0      1       2      3
                xs_ = xs.transpose((0, 3, 1, 2))
                ys_ = ys.transpose((0, 3, 1, 2))

                yield torch.from_numpy(xs_).float(), torch.from_numpy(ys_).float()

            else:
                # for keras

                yield xs, ys




# Data Loader interface

class DataLoader_from_generator():
    def __init__(self, *args, **kwargs):
        self.gen = data_generator(*args, **kwargs)

        bs = args[0]
        file_list = args[1]
        self.steps = len(file_list) // bs

    def __iter__(self):
        for i in range(self.steps):
            yield next(self.gen)

    def __len__(self):
        return self.steps


class DataLoader_from_np_generator():
    def __init__(self, *args, **kwargs):
        self.gen = np_data_generator(*args, **kwargs)

        bs = args[0]
        file_list = args[1]
        self.steps = len(file_list) // bs

    def __iter__(self):
        for i in range(self.steps):
            yield next(self.gen)

    def __len__(self):
        return self.steps


class DataLoader_from_img_to_img_generator():
    def __init__(self, *args, **kwargs):
        self.gen = img_to_img_data_generator(*args, **kwargs)

        bs = args[0]
        file_list = args[1]
        self.steps = len(file_list) // bs

    def __iter__(self):
        for i in range(self.steps):
            yield next(self.gen)

    def __len__(self):
        return self.steps
