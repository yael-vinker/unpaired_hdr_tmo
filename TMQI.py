import numpy as np
import imageio
import matplotlib.pyplot as plt
import skimage
from scipy import signal
from scipy.signal import convolve
from skimage.util import view_as_blocks
from scipy.stats import norm, beta
import hdr_image_utils

def _RGBtoY(RGB):
    M = np.asarray([[0.2126, 0.7152, 0.0722], ])
    Y = np.dot(RGB.reshape(-1, 3), M.T)
    return Y.reshape(RGB.shape[0:2])

def show_im(im, isTensor=False):
    if isTensor:
        im = im.clone().permute(1, 2, 0).detach().cpu().numpy()
        gray_im = np.squeeze(im)
        print(gray_im.shape)
        plt.imshow(gray_im, cmap='gray')
        plt.show()
    else:
        plt.imshow(im)
        plt.show()

def StatisticalNaturalness(L_ldr, win=11):
    phat1 = 4.4
    phat2 = 10.1
    muhat = 115.94
    sigmahat = 27.99
    u = np.mean(L_ldr)

    # moving window standard deviation using reflected image
    # if self.original:
    W, H = L_ldr.shape
    w_extra = (11 - W % 11)
    h_extra = (11 - H % 11)
    # zero padding to simulate matlab's behaviour
    if w_extra > 0 or h_extra > 0:
        test = np.pad(L_ldr, pad_width=((0, w_extra), (0, h_extra)), mode='constant')
    else:
        test = L_ldr
    # block view with fixed block size, like in the original article
    view = view_as_blocks(test, block_shape=(11, 11))
    sig = np.mean(np.std(view, axis=(-1, -2)))
    # else:
    #     # deviation: moving window with reflected borders
    #     sig = np.mean(generic_filter(L_ldr, np.std, size=win))

    beta_mode = (phat1 - 1.) / (phat1 + phat2 - 2.)
    C_0 = beta.pdf(beta_mode, phat1, phat2)
    C = beta.pdf(sig / 64.29, phat1, phat2)
    pc = C / C_0
    B = norm.pdf(u, muhat, sigmahat)
    B_0 = norm.pdf(muhat, muhat, sigmahat)
    pb = B / B_0
    N = pb * pc
    return N

def to_0_1_range(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def _Slocal(img1, img2, window, sf, C1=0.01, C2=10.):

    window = window / window.sum()

    mu1 = convolve(window, img1, 'valid')
    mu2 = convolve(window, img2, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = convolve(img2 * img2, window, 'valid') - mu2_sq

    sigma1 = np.sqrt(np.maximum(sigma1_sq, 0))
    sigma2 = np.sqrt(np.maximum(sigma2_sq, 0))

    sigma12 = convolve(img1 * img2, window, 'valid') - mu1_mu2

    CSF = 100.0 * 2.6 * (0.0192 + 0.114 * sf) * np.exp(- (0.114 * sf) ** 1.1)
    u_hdr = 128 / (1.4 * CSF)
    sig_hdr = u_hdr / 3.

    sigma1p = norm.cdf(sigma1, loc=u_hdr, scale=sig_hdr)

    u_ldr = u_hdr
    sig_ldr = u_ldr / 3.

    sigma2p = norm.cdf(sigma2, loc=u_ldr, scale=sig_ldr)

    s_map = ((2 * sigma1p * sigma2p + C1) / (sigma1p**2 + sigma2p**2 + C1)
             * ((sigma12 + C2) / (sigma1 * sigma2 + C2)))
    s = np.mean(s_map)
    return s, s_map

def _StructuralFidelity(L_hdr, L_ldr, level, weight, window):

    f = 32
    s_local = []
    s_maps = []
    kernel = np.ones((2, 2)) / 4.0

    for _ in range(level):
        f = f / 2
        sl, sm = _Slocal(L_hdr, L_ldr, window, f)

        s_local.append(sl)
        s_maps.append(sm)

        # averaging
        filtered_im1 = convolve(L_hdr, kernel, mode='valid')
        filtered_im2 = convolve(L_ldr, kernel, mode='valid')

        # downsampling
        L_hdr = filtered_im1[::2, ::2]
        L_ldr = filtered_im2[::2, ::2]

    S = np.prod(np.power(s_local, weight))
    return S, s_local, s_maps

def print_result(Q, S, N, s_local, s_maps):
    from scipy.misc import imsave
    # for idx, sm in enumerate(s_maps):
    #     filename = "%s%i.%s" % ("s_map_", idx + 1, "float32")
    #
    #     try:
    #         out = sm.astype("float32")
    #         out.tofile(filename)
    #     except TypeError:
    #         imsave(filename, sm)
    print("Q = ", Q, " S = ", S, " N = ", N, "s_local = ", s_local)

# def TMQI(ldr, hdr):
#


if __name__ == '__main__':
    im_hdr = imageio.imread("data/hdr_data/hdr_data/S0010.hdr").astype('float32')
    L_hdr = skimage.transform.resize(im_hdr, (256, 256),  mode='reflect', preserve_range=False)
    hdr_image_utils.print_image_details(L_hdr, "L_hdr AFTER READ")

    data = np.load("data/hdr_log_data/hdr_log_data/S0010_1000.npy", allow_pickle=True)
    L_ldr = data[()]["input_image"]
    L_ldr = np.squeeze(L_ldr.clone().permute(1, 2, 0).detach().cpu().numpy())
    L_ldr = to_0_1_range(L_ldr)
    hdr_image_utils.print_image_details(L_ldr, "L_ldr AFTER READ")
    L_hdr = L_ldr

    if len(L_hdr.shape) == 3:
        # Processing RGB images
        L_hdr = _RGBtoY(L_hdr)

    if len(L_ldr.shape) == 3:
        L_ldr = _RGBtoY(L_ldr)

    hdr_image_utils.print_image_details(L_hdr, "L_hdr AFTER GRAY SCALE")

    a = 0.8012
    Alpha = 0.3046
    Beta = 0.7088
    lvl = 5  # levels
    weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    M, N = L_hdr.shape
    window = None

    if window is None:
        gauss = signal.gaussian(11, 1.5)
        window = np.outer(gauss, gauss)

    # Naturalness should be calculated before rescaling
    N = StatisticalNaturalness(L_ldr)

    # The images should have the same dynamic ranges, e.g. [0,255]
    factor = float(2**8 - 1.)

    hdr_image_utils.print_image_details(L_ldr, "L_ldr before RE-FACTOR")
    hdr_image_utils.print_image_details(L_hdr, "L_hdr before RE-FACTOR")

    # if self.original:
    L_hdr = factor * (L_hdr - L_hdr.min()) / (L_hdr.max() - L_hdr.min())
    L_ldr = factor * (L_ldr - L_ldr.min()) / (L_ldr.max() - L_ldr.min())

    hdr_image_utils.print_image_details(L_ldr, "L_ldr AFTER RE-FACTOR")
    hdr_image_utils.print_image_details(L_hdr, "L_hdr AFTER RE-FACTOR")

    S, s_local, s_maps = _StructuralFidelity(L_hdr, L_ldr, lvl, weight, window)
    Q = a * (S ** Alpha) + (1. - a) * (N ** Beta)
    print_result(Q, S, N, s_local, s_maps)

    # else:
    #     # but we really should scale them similarly...
    #     L_hdr = factor * (L_hdr - L_hdr.min()) / (L_hdr.max() - L_hdr.min())
    #     L_ldr = factor * (L_ldr - L_ldr.min()) / (L_ldr.max() - L_ldr.min())
    #

