import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import hdr_image_utils

def save_text_to_image(output_path, text):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (256, 256), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((5,5), text, fill=(0,0,0))
    img.save(os.path.join(output_path, "all.png"))

def save_single_tone_mapped_result(output_path, title, method_name, image):
    plt.figure(figsize=(30,30))
    plt.axis("off")
    plt.title(title, fontsize=15)
    plt.imshow(image)
    plt.savefig(os.path.join(output_path, method_name))
    plt.close()
#
# def ours(original_im, net_path):
#     import torch
#     device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
#     G_net = Unet.UNet(1, 1, 0, bilinear=False).to(device)
#     checkpoint = torch.load(net_path)
#     state_dict = checkpoint['modelG_state_dict']
#     # G_net.load_state_dict(checkpoint['modelG_state_dict'])
#
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[7:]  # remove `module.`
#         new_state_dict[name] = v
#     # load params
#     G_net.load_state_dict(new_state_dict)
#     G_net.eval()
#     data = np.load("data/hdr_log_data/hdr_log_data/belgium_1000.npy", allow_pickle=True)
#     L_hdr_log = data[()]["input_image"].to(device)
#     inputs = L_hdr_log.unsqueeze(0)
#     # outputs = net(inputs)
#     # L_ldr = np.squeeze(L_ldr.clone().permute(1, 2, 0).detach().cpu().numpy())
#     # _L_ldr = to_0_1_range(L_ldr)
#     ours_tone_map = torch.squeeze(G_net(inputs), dim=0)
#     # ours_tone_map = np.squeeze(ours_tone_map, axis=0)
#     return back_to_color(original_im, ours_tone_map.clone().permute(1, 2, 0).detach().cpu().numpy())

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
    im = log_1000(im)
    return exp_map(im)

def log_100_exp(im):
    im = log_100(im)
    return exp_map(im)

def exp_map(im):
    import math
    im_0_1 = (im - np.min(im)) / (np.max(im) - np.min(im))
    # im_lm = im_0_1 * self.log_factor
    im_exp = np.exp(im_0_1) - 1
    im_end = im_exp / (math.exp(1) - 1)
    return im_end

def Dargo_tone_map(im):
    # im = im / np.max(im)
    # Tonemap using Drago's method to obtain 24-bit color image
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7, 0.85)
    ldrDrago = tonemapDrago.process(im)
    # ldrDrago = 3 * ldrDrago
    # hdr_image_utils.print_image_details(ldrDrago,"dargo")
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


def calculate_TMQI_results_for_selected_methods(input_path, output_path):
    import operator
    import skimage
    import TMQI
    for img_name in os.listdir(input_path):
        hdr_path = os.path.join(input_path, img_name)
        im_hdr_original = hdr_image_utils.read_hdr_image(hdr_path)
        im_hdr_original = skimage.transform.resize(im_hdr_original, (int(im_hdr_original.shape[0] / 3),
                                                                     int(im_hdr_original.shape[1] / 3)), mode='reflect',
                                                   preserve_range=False).astype("float32")
        new_output_path = os.path.join(output_path, os.path.splitext(img_name)[0])
        if not os.path.exists(new_output_path):
            os.makedirs(new_output_path)
        tone_map_methods = {"Reinhard": Reinhard_tone_map,
                            "Dargo": Dargo_tone_map,
                            "Mantiuk": Mantiuk_tone_map,
                            "Durand": Durand_tone_map,
                            "log100_exp": log_100_exp,
                            "log1000_exp": log_1000_exp,
                            "log100": log_100,
                            "log1000": log_1000,
                            "log": log_1}
        methods_and_Q_results = {}
        for method_name in tone_map_methods.keys():
            tone_mapped_result = tone_map_methods[method_name](im_hdr_original)
            Q, S, N = TMQI.TMQI(im_hdr_original, tone_mapped_result)
            methods_and_Q_results[method_name] = Q
            title = method_name + "\nQ = " + str(Q) + "\nS = " + str(S) + "\n" + "N = " + str(N) + "\n" + "max = " \
                    + str(np.max(tone_mapped_result)) + "   min = " + str(np.min(tone_mapped_result))
            save_single_tone_mapped_result(new_output_path, title, method_name, tone_mapped_result)

        sorted_methods_and_Q_results_by_Q = sorted(methods_and_Q_results.items(), key=operator.itemgetter(1))[::-1]
        text = ""
        for method_and_q_tuple in sorted_methods_and_Q_results_by_Q:
            subtitle = method_and_q_tuple[0] + " : " + str(method_and_q_tuple[1]) + "\n"
            text += subtitle
        save_text_to_image(new_output_path, text)

