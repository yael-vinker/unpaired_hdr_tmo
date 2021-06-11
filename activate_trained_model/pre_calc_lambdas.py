import scipy.optimize as optimize
import numpy as np
import os
import utils.hdr_image_util as hdr_image_util
import matplotlib.pyplot as plt


def cross_entropy(factor, gray_im, targets, bins_):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    gray_im_log = np.log10(gray_im * factor + 1)
    gray_im_log = gray_im_log / gray_im_log.max()
    gray_im_flat = np.reshape(gray_im_log, (-1,))
    predictions, all_bins = np.histogram(gray_im_flat, bins=bins_, density=True, range=(0, 1))
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def verify_lambda_dict(args):
    f_factor_path = os.path.join(args.f_factor_path)
    if not os.path.isfile(f_factor_path):
        return False
    data = np.load(f_factor_path, allow_pickle=True)[()]
    for im_name in os.listdir(args.input_images_path):
        if os.path.splitext(im_name)[0] not in data:
            return False
    return True


def calc_lambda(args, extensions):
    if verify_lambda_dict(args):
        return
    print("Calculating lambdas for input data...")
    input_names = os.listdir(args.input_images_path)
    mean_data = np.load(args.mean_hist_path, allow_pickle=True)[()]
    targets, all_bins = mean_data["mean_vals"], mean_data["all_bins"]
    res_dict = {}
    lambdas_output_path = os.path.join(args.lambda_output_path, "input_images_lambdas.npy")
    if os.path.isfile(lambdas_output_path):
        res_dict = np.load(lambdas_output_path, allow_pickle=True)[()]

    for img_name in input_names:
        if os.path.splitext(img_name)[0] not in res_dict and os.path.splitext(img_name)[1] in extensions:
            im_path = os.path.join(args.input_images_path, img_name)
            rgb_img = hdr_image_util.read_hdr_image(im_path)
            gray_im = hdr_image_util.to_gray(rgb_img)
            if gray_im.min() < 0:
                gray_im = gray_im - gray_im.min()
            gray_im_ = hdr_image_util.reshape_image(gray_im, train_reshape=False)
            gray_im_ = gray_im_ / gray_im_.max()
            sol = optimize.differential_evolution(cross_entropy, args=(gray_im_, targets, args.bins),
                                                  bounds=[(1, 1000000000)], maxiter=1000)
            print("[%s] [%.4f] [%.4f]" % (img_name, sol.x, sol.fun))
            res_dict[os.path.splitext(img_name)[0]] = sol.x[0]
            np.save(lambdas_output_path, res_dict)
    args.f_factor_path = lambdas_output_path
    print("Lambdas data saved successfully")


def plot_res(lambda_factor, gray_im):
    plt.figure(figsize=(14, 8))
    gray_im_ = gray_im - gray_im.min()
    gray_im_ = gray_im_ / gray_im_.max()

    plt.subplot(2, 3, 1)
    plt.title("input")
    plt.imshow(gray_im, cmap='gray')
    plt.axis("off")

    gray_im_flat = np.reshape(gray_im, (-1,))
    predictions, all_bins = np.histogram(gray_im_flat, bins=20, density=True)#, range=(0, 1))
    plt.subplot(2, 3, 4)
    plt.bar(all_bins[:-1] + np.diff(all_bins) / 2, predictions, np.diff(all_bins), align='center', alpha=0.5)

    gray_im_log = np.log10(gray_im_ * 1000 * 255 * 0.1 + 1)
    gray_im_log = gray_im_log / gray_im_log.max()
    plt.subplot(2, 3, 2)
    plt.title("lambda 1000")
    plt.imshow(gray_im_log, cmap='gray')
    plt.axis("off")

    gray_im_flat = np.reshape(gray_im_log, (-1,))
    predictions, all_bins = np.histogram(gray_im_flat, bins=20, density=True)#, range=(0, 1))
    plt.subplot(2, 3, 5)
    plt.bar(all_bins[:-1] + np.diff(all_bins) / 2, predictions, np.diff(all_bins), align='center', alpha=0.5)


    gray_im_log = np.log10(gray_im_ * lambda_factor * 255 * 0.1 + 1)
    gray_im_log = gray_im_log / gray_im_log.max()
    plt.subplot(2, 3, 3)
    plt.title("lambda %.2f" % lambda_factor)
    plt.imshow(gray_im_log, cmap='gray')
    plt.axis("off")

    gray_im_flat = np.reshape(gray_im_log, (-1,))
    predictions, all_bins = np.histogram(gray_im_flat, bins=20, density=True)#, range=(0, 1))
    plt.subplot(2, 3, 6)
    plt.bar(all_bins[:-1] + np.diff(all_bins) / 2, predictions, np.diff(all_bins), align='center', alpha=0.5)


def plot_data_for_ablation(mean_hist_path, f_factor_path, input_images_path, image_name):
    mean_data = np.load(mean_hist_path, allow_pickle=True)[()]
    targets, all_bins = mean_data["mean_vals"], mean_data["all_bins"]
    plt.figure()
    plt.bar(all_bins[:-1] + np.diff(all_bins) / 2, targets, width=np.diff(all_bins), align='center', alpha=0.5)
    plt.title("DIV2K mean histogram")
    plt.savefig("/Users/yaelvinker/Documents/university/lab/ECCV/lambda_process/plots/mean_hist.jpg")
    plt.show()

    f_factor_dict = np.load(f_factor_path, allow_pickle=True)[()]
    gray_im = hdr_image_util.to_gray(hdr_image_util.read_hdr_image(os.path.join(input_images_path, image_name + ".exr")))
    plot_res(f_factor_dict[image_name], gray_im)
    plt.savefig("/Users/yaelvinker/Documents/university/lab/ECCV/lambda_process/plots/" + image_name + ".jpg")
    plt.show()

#
# f_factor_path="/Users/yaelvinker/Documents/university/lab/lum_hist_re/valid_hist_dict_20_bins.npy"
# mean_hist_path = "/Users/yaelvinker/PycharmProjects/lab/activate_trained_model/lambda_data/ldr_avg_hist_900_images_20_bins.npy"
# input_images_path = "/Users/yaelvinker/Documents/university/lab/ECCV/lambda_process/input_images_for_comparison"
# image_name = "507"
# plot_data_for_ablation(mean_hist_path, f_factor_path, input_images_path, image_name)