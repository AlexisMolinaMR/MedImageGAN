import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import trange

from .inception_net import InceptionV3


def get_statistics(images, model, device, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : List of image
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, DIM) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(images):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(images)

    acts = np.empty((len(images), DIM))

    if verbose:
        iterator = trange(0, len(images), batch_size, dynamic_ncols=True)
    else:
        iterator = range(0, len(images), batch_size)

    for start in iterator:
        end = start + batch_size
        batch_images = images[start: end]

        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal DIM.
        if len(pred.shape) != 2 and (pred.size(2) != 1 or pred.size(3) != 1):
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        acts[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)

    return mu, sigma


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

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def get_fid_score(images, stats_cache, device, batch_size=50, verbose=False):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[DIM]
    model = InceptionV3([block_idx]).to(device)

    f = np.load(stats_cache)
    m1, s1 = get_statistics(images, model, device, batch_size, verbose)
    m2, s2 = f['mu'][:], f['sigma'][:]
    f.close()
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def get_inception_score(images, device, splits=10, batch_size=32,
                        verbose=False):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    preds = []
    if verbose:
        iterator = trange(0, len(images), batch_size, dynamic_ncols=True)
    else:
        iterator = range(0, len(images), batch_size)
    for start in iterator:
        end = start + batch_size
        batch_images = images[start: end]
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        pred = model(batch_images)[0]
        preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[
            (i * preds.shape[0] // splits):
            ((i + 1) * preds.shape[0] // splits), :]
        kl = part * (
            np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


def get_inception_and_fid_score(images, device, fid_cache, is_splits=10,
                                batch_size=50, verbose=False):
    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx1, block_idx2]).to(device)
    model.eval()

    if batch_size > len(images):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(images)

    fid_acts = np.empty((len(images), 2048))
    is_probs = np.empty((len(images), 1008))

    if verbose:
        iterator = trange(0, len(images), batch_size)
    else:
        iterator = range(0, len(images), batch_size)

    for start in iterator:
        end = start + batch_size
        batch_images = images[start: end]

        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
        fid_acts[start: end] = pred[0].view(-1, 2048).cpu().numpy()
        is_probs[start: end] = pred[1].cpu().numpy()

    # Inception Score
    scores = []
    for i in range(is_splits):
        part = is_probs[
            (i * is_probs.shape[0] // is_splits):
            ((i + 1) * is_probs.shape[0] // is_splits), :]
        kl = part * (
            np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    is_score = (np.mean(scores), np.std(scores))

    # FID Score
    m1 = np.mean(fid_acts, axis=0)
    s1 = np.cov(fid_acts, rowvar=False)
    f = np.load(fid_cache)
    m2, s2 = 0.5, 0.5
    #m2, s2 = 0.4, 0.3
    f.close()
    fid_score = calculate_frechet_distance(m1, s1, m2, s2)

    return is_score, fid_score
