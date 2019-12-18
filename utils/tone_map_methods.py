import numpy as np
import cv2

def log_1(im, range_factor=1):
    max_origin = np.max(im)
    image_new_range = (im / max_origin) * range_factor
    im_log = np.log(image_new_range + 1)
    im = (im_log / np.log(range_factor + 1)).astype('float32')
    return im

def log_100(im, range_factor=100):
    max_origin = np.max(im)
    image_new_range = (im / max_origin) * range_factor
    im_log = np.log(image_new_range + 1)
    im = (im_log / np.log(range_factor + 1)).astype('float32')
    return im

def log_1000(im, range_factor=1000):
    max_origin = np.max(im)
    image_new_range = (im / max_origin) * range_factor
    im_log = np.log(image_new_range + 1)
    im = (im_log / np.log(range_factor + 1)).astype('float32')
    return im

def log_1000_exp(im):
    im_log_normalize_tensor = apply_preproccess_for_hdr_im(im)
    log1000_im = net_G_pipeline(im_log_normalize_tensor, im)
    return log1000_im

def log_100_exp(im):
    im = log_100(im)
    return exp_map(im)

def exp_map(im):
    import math
    im_0_1 = (im - np.min(im)) / (np.max(im) - np.min(im))
    im_exp = np.exp(im_0_1) - 1
    im_end = im_exp / (math.exp(1) - 1)
    return im_end

def Dargo_tone_map(im):
    # Tonemap using Drago's method to obtain 24-bit color image
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7, 0.85)
    ldrDrago = tonemapDrago.process(im)
    return ldrDrago


def Durand_tone_map(im):
    # Tonemap using Durand's method obtain 24-bit color image
    tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
    ldrDurand = tonemapDurand.process(im)
    # ldrDurand = 3 * ldrDurand
    # hdr_image_utils.print_image_details(ldrDurand,"durand")
    return ldrDurand

def Reinhard_tone_map(im):
    # Tonemap using Reinhard's method to obtain 24-bit color image
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(im)
    return ldrReinhard

def Mantiuk_tone_map(im):
    # Tonemap using Mantiuk's method to obtain 24-bit color image
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(im)
    return ldrMantiuk

def Durand_tone_map(im):
    # Tonemap using Durand's method obtain 24-bit color image
    tonemapDurand = cv2.createTonemapDurand(1.5,4,1.0,1,1)
    ldrDurand = tonemapDurand.process(im)
    return ldrDurand