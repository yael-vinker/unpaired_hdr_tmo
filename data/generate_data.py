import numpy as np
import matplotlib.pyplot as plt
import random
import os
from skimage import draw
import cv2
import imageio
from PIL import Image

def create_real_data(path, num_of_images):
    for i in range(num_of_images):
        black_background = np.zeros((256, 256, 3))
        white_squer = np.ones((28, 28, 3))
        random_y = random.randint(0, 256 - 28)
        random_x = random.randint(0, 256 - 28)
        black_background[random_y: random_y + 28, random_x: random_x + 28] = white_squer
        save_image(black_background, os.path.join(path, "im" + str(i) + ".jpg"))
        # fig = plt.figure()
        # fig.subplots_adjust(bottom=0)
        # fig.subplots_adjust(top=1)
        # fig.subplots_adjust(right=1)
        # fig.subplots_adjust(left=0)
        # # ax = fig.add_subplot(1, 1, 1)
        # plt.axis('off')
        # plt.imshow(black_background)
        # # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # # plt.savefig(os.path.join(path, "im" + str(i)), bbox_inches=extent)
        # # plt.figure()
        # plt.savefig(os.path.join(path, "im" + str(i)))
        # plt.close()

def create_fake_data(path, num_of_images):
    for i in range(num_of_images):
        black_background = np.zeros((256, 256, 3))
        rr, cc = draw.circle(128, 128, radius=40, shape=black_background.shape)
        black_background[rr, cc] = 1
        save_image(black_background, os.path.join(path, "im" + str(i) + ".jpg"))

def test(path):
    for img_name in os.listdir(path):
        im_path = os.path.join(path, img_name)
        image = cv2.imread(im_path)
        print("cv2 ",image.shape)
        im_origin = imageio.imread(im_path)
        print("imageio ",im_origin.shape)



def save_image(data, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(fn, dpi=height)
    plt.close()

if __name__ == '__main__':
    real_train_data_path = os.path.join("check_data", "train_real", "train_real")
    create_real_data(real_train_data_path, 8)

    real_test_data_path = os.path.join("check_data", "test_real", "test_real")
    create_real_data(real_test_data_path, 4)

    fake_train_data_path = os.path.join("check_data", "train_fake", "train_fake")
    create_fake_data(fake_train_data_path, 8)

    fake_test_data_path = os.path.join("check_data", "test_fake", "test_fake")
    create_fake_data(fake_test_data_path, 4)

    test(real_train_data_path)
    test(real_test_data_path)
    test(fake_train_data_path)
    test(fake_test_data_path)



