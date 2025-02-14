import scipy.signal as signal
import matplotlib.pyplot as plt
import skimage.util
import scipy.ndimage as ndimage
from skimage.color import rgb2gray
from skimage.io import imread, imshow
import numpy as np
import matplotlib



plt.gray()
lena= rgb2gray(imread('lena.tiff'))
cameraman = imread('cameraman.tif').astype(np.float64) / 255

# imshow(lena)
# plt.show()

# imshow(cameraman)
# plt.show()


def gaussian_filter(n_rows, n_cols, stdv):
    """
    Returns a 2d Gaussian image filter.
    """
    g_r = signal.windows.gaussian(n_rows, stdv)
    g_c = signal.windows.gaussian(n_cols, stdv)

    G = np.outer(g_r, g_c)

    return G/np.sum(G)

def PSNR(f # original image
        ,g # edited image
        ):
    return 10*np.log10(1.0/ np.mean(np.square(f-g)))

def discrete_convolution(image):
    h1 = (1 / 6) * np.ones((1 , 6))  # [1/6, 1/6, 1/6 ... ]
    h2 = h1.T
    h3 = np.array([[-1, 1]])

    imshow(image)
    plt.show()

    image_convolve = signal.convolve2d(image, h1)
    imshow(image_convolve, vmin=0, vmax=1)
    plt.show()

    image_convolve = signal.convolve2d(image, h2)
    imshow(image_convolve, vmin=0, vmax=1)
    plt.show()

    image_convolve = signal.convolve2d(image, h3)
    imshow(image_convolve, vmin=0, vmax=1)
    plt.show()


def noise_gen():
    f = np.hstack([0.3 * np.ones((200, 100)), 0.7 * np.ones((200, 100))])
    imshow(f, vmin=0, vmax=1)
    plt.show()

    plt.hist(f.flatten())
    plt.show()

    gaussian = skimage.util.random_noise(f, mode='gaussian')
    imshow(gaussian, vmin=0, vmax=1)
    plt.show()

    plt.hist(gaussian.flatten())
    plt.show()

    sp = skimage.util.random_noise(f, mode='s&p')
    imshow(sp, vmin=0, vmax=1)
    plt.show()

    plt.hist(sp.flatten())
    plt.show()

    speckle = skimage.util.random_noise(f, mode='speckle')
    imshow(speckle, vmin=0, vmax=1)
    plt.show()

    plt.hist(speckle.flatten())
    plt.show()


def noise_redux(image):
    imshow(image)
    plt.show()

    plt.hist(image.flatten())
    plt.show()

    # image_gaus = skimage.util.random_noise(image, mode='gaussian', var=0.002)
    # imshow(image_gaus)
    # plt.show()

    # plt.hist(image_gaus.flatten())
    # plt.show()

    # print(PSNR(image_gaus, image))



    # avg_filter_kernel = np.ones((3, 3)) / (3.0 * 3.0)
    # imshow(avg_filter_kernel)
    # plt.show()

    # image_convolve = ndimage.convolve(image_gaus, avg_filter_kernel)
    # imshow(image_convolve)
    # plt.show()

    # plt.hist(image_convolve.flatten())
    # plt.show()

    # print(PSNR(image, image_convolve))



    # gaus_filter = gaussian_filter(7, 7, 1)
    # imshow(gaus_filter)
    # plt.show()

    # image_convolve = ndimage.convolve(image_gaus, gaus_filter)
    # imshow(image_convolve)
    # plt.show()

    # plt.hist(image_convolve.flatten())
    # plt.show()

    # print(PSNR(image, image_convolve))



    image_sp = skimage.util.random_noise(image, mode='s&p')
    # imshow(image_sp)
    # plt.show()

    # plt.hist(image_sp.flatten())
    # plt.show()

    avg_filter_kernel = np.ones((7, 7)) / (7.0 * 7.0)

    # gaus_filter = gaussian_filter(7, 7, 1)

    image_convolve = ndimage.convolve(image_sp, avg_filter_kernel)
    PSNR(image_sp, image_convolve)
    # imshow(image_convolve)
    # plt.show()

    plt.hist(image_convolve.flatten())
    plt.show()

    print(PSNR(image, image_convolve))

    # image_convolve = ndimage.convolve(image_sp, gaus_filter)
    # imshow(image_convolve)
    # plt.show()

    # plt.hist(image_convolve.flatten())
    # plt.show()

    # print(PSNR(image, image_convolve))


    # median_filter = ndimage.median_filter(image_sp, size=3)
    # imshow(median_filter)
    # plt.show()

    # plt.hist(median_filter.flatten())
    # plt.show()

    # print(PSNR(image, median_filter))


def sharpen_in_spatial_domain(image):
    imshow(image)
    plt.show()

    gaussian_filter7x7 = gaussian_filter(7, 7, 1)
    image_gaus_filt = ndimage.convolve(image, gaussian_filter7x7)

    # Gaussian filtered image
    imshow(image_gaus_filt)
    plt.show()

    # image - gaussian filtered image
    edge_image = image - image_gaus_filt
    imshow(edge_image)
    plt.show()

    # image + edges of image
    image_plus_edge_image = image + edge_image
    imshow(image_plus_edge_image)
    plt.show()

    # image + half edges of image
    image_plus_half_edges = image + 0.5*edge_image
    imshow(image_plus_half_edges)
    plt.show()




def main():
    # discrete_convolution(lena)
    # noise_gen()
    noise_redux(lena)
    # sharpen_in_spatial_domain(cameraman)



if __name__ == '__main__':
    main()
