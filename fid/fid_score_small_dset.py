#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import sys
import inspect
import os
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image
import params as params
# from skimage.transform import resize

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3


def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename).resize((299, 299), Image.BICUBIC), dtype=np.uint8)[..., :3]
    # return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


# def get_batch(files, batch_size, start, end, device, noise_type, a, apply_transforms):
#     images = np.zeros([batch_size, 64, 64, 3])
#     for f, i in zip(files[start:end], range(batch_size)):
#         im = imread(str(f)).astype('uint8')
#         cropped_image = np.array(Image.fromarray(im).resize((108, 108)))
#         cropped_image = np.array(Image.fromarray(cropped_image[22:86, 22:86]).resize((64, 64))).astype("float32")
#         cropped_image = cropped_image / 127.5 - 1
#         if apply_transforms:
#             transformed_im = apply_transform(cropped_image, noise_type, a)
#             plt.imshow(transformed_im.astype('uint8'))
#             plt.show()
#         else:
#             transformed_im = (cropped_image + 1.) * 127.5
#         if i == 0:
#             print("#  -- Range of cropped images: [%.2f, %.2f]" % (cropped_image.min(), cropped_image.max()))
#             print("#  -- Range of transformed images: [%.2f, %.2f]" % (transformed_im.min(), transformed_im.max()))
#         images[i] = transformed_im
#     # Reshape to (n_images, 3, height, width)
#     images = images.transpose((0, 3, 1, 2))
#     # images /= 255
#
#     batch = torch.from_numpy(images).type(torch.FloatTensor)
#     batch = batch.to(device)
#     return batch

def get_batch2(files, batch_size, start, end, files_format):
    images = np.zeros([batch_size, 3, params.input_size, params.input_size])
    for f, i in zip(files[start:end], range(batch_size)):
        data = np.load(f, allow_pickle=True)
        color_im = data[()]["display_image"]
        images[i] = color_im
    # Reshape to (n_images, 3, height, width)
    # images = images.transpose((0, 3, 1, 2))
    images /= 255

    batch = torch.from_numpy(images).type(torch.FloatTensor)
    if torch.cuda.is_available():
        batch = batch.cuda()
    return batch


def get_batch(files, batch_size, start, end, files_format):
    import PIL
    # images = np.array([imread(str(f)).astype(np.float32) for f in files[start:end]])
    images = np.zeros([batch_size,  params.input_size, params.input_size, 3])
    for f, i in zip(files[start:end], range(batch_size)):
        im = imread(str(f)).astype(np.float32)
        # im = im.resize((299, 299), PIL.Image.BICUBIC)
        # im = np.array(Image.fromarray(im).resize((299, 299)))
        # im = resize(im, (299, 299), mode='reflect', preserve_range=False).astype("float32")
        print("max",im.max())
        print("min",im.min())
        images[i] = im
    # Reshape to (n_images, 3, height, width)
    # images = images.transpose((0, 3, 1, 2))
    images = images.transpose((0, 3, 1, 2))
    images /= 255

    batch = torch.from_numpy(images).type(torch.FloatTensor)
    if torch.cuda.is_available():
        batch = batch.cuda()
    return batch


def get_activations_for_small_dataset(files, model, batch_size=50, dims=2048,
                    device="cpu", verbose=False, files_format="jpg"):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)
    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size * 64

    pred_arr = np.empty((n_used_imgs, dims))

    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start_files = i * batch_size
        end_files = start_files + (batch_size * 64)
        start = i * batch_size * 64
        end = start + (batch_size * 64)

        batch = get_batch(files, batch_size, start_files, end_files, files_format)
        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 8 or pred.shape[3] != 8:
            pred = adaptive_avg_pool2d(pred, output_size=(8, 8))
        permuted_pred = pred.permute((0, 2, 3, 1))
        reshaped_pred = permuted_pred.cpu().data.numpy().reshape(batch_size * 64, dims)
        pred_arr[start:end] = reshaped_pred
    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50,
                                    dims=2048, device="", verbose=False, files_format="jpg"):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations_for_small_dataset(files, model, batch_size, dims, device, verbose, files_format)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, device, im_from, im_to):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files_jpg = list(path.glob('*.jpg'))[im_from:im_to]
        files = files_jpg
        files_format = "jpg"
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, False, files_format)

    return m, s


def calculate_fid_given_paths(path_real, path_fake, batch_size, device, dims, number_of_images):
    """Calculates the FID of two paths"""

    if not os.path.exists(path_real):
        raise RuntimeError('Invalid path: %s' % path_real)
    if not os.path.exists(path_fake):
        raise RuntimeError('Invalid path: %s' % path_fake)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = _compute_statistics_of_path(path_real, model, batch_size,
                                         dims, device, im_from=0, im_to=number_of_images)
    m2, s2 = _compute_statistics_of_path(path_fake, model, batch_size,
                                         dims, device, im_from=0, im_to=number_of_images)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    # calc FID
    return fid_value


def apply_preprocess_folder(path_fake):
    from utils import hdr_image_util
    import tranforms as transforms_
    path = pathlib.Path(path_fake)
    output_path = os.path.join(path, "npy_version")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    files = list(path.glob('*.jpg'))
    for f in files:
        color_im = imread(str(f)).astype('uint8')
        color_im = (color_im / color_im.max())
        color_im = hdr_image_util.reshape_image(color_im, train_reshape=True)
        im = (color_im * 255).astype('uint8')
        im = transforms_.image_transform_no_norm(im)
        cur_output_path = os.path.join(output_path, os.path.os.path.splitext(os.path.basename(f))[0] + ".npy")
        data = {'display_image': im}
        np.save(cur_output_path, data)
    return output_path


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_real', type=str,
                        help=('Path to the generated images or '
                              'to .npz statistic files'))
    parser.add_argument('--path_fake', type=str,
                        help=('Path to the generated images or '
                              'to .npz statistic files'))
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('-c', '--gpu', default='', type=str,
                        help='GPU to use (leave blank for CPU only)')
    parser.add_argument('--number_of_images', type=int)
    parser.add_argument('--format', type=str, default="jpg")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    npy_path_fake, npy_path_real = args.path_fake, args.path_real
    fid_value = calculate_fid_given_paths(npy_path_real,
                                          npy_path_fake,
                                              args.batch_size,
                                              device,
                                              args.dims,
                                              args.number_of_images)
    print()
    print(npy_path_real)
    print(npy_path_fake)
    print('FID: ', fid_value)

