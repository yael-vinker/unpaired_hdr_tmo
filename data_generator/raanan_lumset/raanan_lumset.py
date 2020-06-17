from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import imageio


fn = 'belgium.npy'
M = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/folders/data/belgium.hdr")
# M = np.load(fn, allow_pickle=True)[()]["display_image"]
# print(M)
# M = M[:, 512 - 170:1024 + 512 + 170, :]

J = np.mean(M, axis=2)
J = np.reshape(J, (-1,))
M = M / np.max(J)
J = J / np.max(J)

npix = J.shape[0]

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
    im.save("output/exp%.2f.jpeg" % C)

    h = np.histogram(I, bins=5)
    h = h[0]

    rat = np.mean(h[0]) / np.mean(h[1])
    print("%d: %.2f (%.2f)" % (i, rat, I.shape[0] / npix))

    if rat > 0.5:
        Cout = C * np.sqrt(2)
        im = M * 255 * Cout
        im[im > 255] = 255
        im = im.astype(np.uint8)
        im = Image.fromarray(im)
        im.save("output/chose_C%.2f.jpg" % Cout)


    plt.hist(I, rwidth=0.9, color='#607c8e', density=True, bins=5)
    plt.box(on=None)

    # plt.plot(h)
    plt.title('r=%.2f, lambda=%.2f' % (rat, C))

    # ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    # plt.grid(axis='y', alpha=0.75)
plt.show()
    # plt.savefig('output/fig_%.2f_%d.png' % (C, i), dpi=300)
    # plt.clf()