from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import imageio
# images = []
# import os
# for p in sorted(os.listdir("/Users/yaelvinker/PycharmProjects/lab/data_generator/raanan_lumset/output")):
#     im = imageio.imread(os.path.join("/Users/yaelvinker/PycharmProjects/lab/data_generator/raanan_lumset/output",p))
#     images.append(im)
# kargs = { 'duration': 0.5 }
# imageio.mimsave('/Users/yaelvinker/PycharmProjects/lab/data_generator/raanan_lumset/movie.gif', images, **kargs)

fn = 'belgium.npy'
M = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data/BarHarborSunrise.hdr")
# M = np.load(fn, allow_pickle=True)[()]["display_image"]
# print(M)
# M = M[:, 512 - 170:1024 + 512 + 170, :]
print(M.shape)
J = np.mean(M, axis=2)
J = np.reshape(J, (-1,))
M = M / np.max(J)
J = J / np.max(J)
J = J[J>0]
npix = J.shape[0]
images = []
# J = J[J!=0]

# print(np.percentile(J, 50))
# print(np.percentile(J, 1))
# print(np.percentile(J, 99))
# print(np.percentile(J, 100))
print(J.min(), J.mean(), J.max())
for i in range(100):

    C = np.sqrt(2) ** i

    I = J * C

    I = I[I < 0.99]

    if I.shape[0] / npix < 0.1:
        break

    im = M * C * 255
    im[im > 255] = 255
    im = im.astype(np.uint8)
    im = Image.fromarray(im)
    plt.imshow(im)
    plt.show()
    # print(I.shape)
    # im.save("output/exp%.2f.jpeg" % C)
    a = np.mean(im, axis=2)
    # plt.subplot(2,1,1)
    # plt.imshow(im)#, cmap='gray')
    # plt.subplot(2, 1, 2)
    # plt.hist(I, rwidth=0.9, color='#607c8e', density=True, bins=5)
    # plt.box(on=None)
    # plt.show()

    h = np.histogram(I, bins=5)
    h = h[0]

    rat = np.mean(h[0]) / np.mean(h[1])
    print(rat)
    print(h[0]/h[1])
    print("%d: %.2f (%.2f)" % (i, rat, I.shape[0] / npix))

    if rat > 0.5:
        Cout = C * np.sqrt(2)
        im = M * 255 * Cout
        im[im > 255] = 255
        im = im.astype(np.uint8)
        im = Image.fromarray(im)
        # im.save("output/chose_C%.2f.jpg" % Cout)
    plt.subplot(2,1,1)
    plt.imshow(im)#, cmap='gray')
    plt.axis("off")
    plt.subplot(2, 1, 2)
    plt.hist(I, rwidth=0.9, color='#607c8e', density=True, bins=5)
    plt.box(on=None)

    # plt.plot(h)
    # plt.title('r=%.2f, lambda=%.2f' % (rat, C))
    plt.title('lambda=%.2f' % (C))
    plt.tight_layout()
    # plt.savefig('output/%02d.png' % (i), dpi=300)
    plt.show()
    plt.clf()
# images = []
# import os
# for p in sorted(os.listdir("/Users/yaelvinker/PycharmProjects/lab/data_generator/raanan_lumset/output")):
#     im = imageio.imread(os.path.join("/Users/yaelvinker/PycharmProjects/lab/data_generator/raanan_lumset/output",p))
#     images.append(im)
# kargs = { 'duration': 5 }
# imageio.mimsave('/Users/yaelvinker/PycharmProjects/lab/data_generator/raanan_lumset/movie.gif', images, **kargs)


    # ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    # plt.grid(axis='y', alpha=0.75)
# plt.show()

    #


# def dr_est(im):
#     """
#
#     :param im: luminance of HDR image in range [0,1]
#     :return:
#     """