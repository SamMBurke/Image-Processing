import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imshow
from skimage.transform import rotate
from skimage.color import rgb2gray
from skimage.morphology import disk, square
from skimage.metrics import peak_signal_noise_ratio as PSNR
import skimage.util
from scipy import signal


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

plt.gray()
lena= rgb2gray(imread('lena.tiff'))
frequnoisy = imread('frequnoisy.tif').astype(np.float64)/255

def fourier_analysis(image):
    test_img = np.zeros((256,256))
    test_img[:,107:148] = 1

    imshow(test_img)
    plt.show()

    four_spec = np.abs(np.fft.fftshift(np.fft.fft2(test_img)))
    imshow(four_spec, cmap='gray')
    plt.show()

    test_img_rotated_45 = rotate(test_img, angle=45)
    imshow(test_img_rotated_45)
    plt.show()

    four_spec_rotated_45 = rotate(four_spec, angle=45)
    imshow(four_spec_rotated_45, cmap='gray')
    plt.show()

    imshow(image)
    plt.show()

    image_amplitude = np.abs(np.fft.fftshift(np.fft.fft2(image)))

    image_phase = (np.fft.fftshift(np.fft.fft2(image)))/image_amplitude

    inv_image_amp = np.log(np.abs(np.fft.ifft2(np.fft.ifftshift(image_amplitude))))
    inv_image_pha = np.log(np.abs(np.fft.ifft2(np.fft.ifftshift(image_phase))))

    imshow(inv_image_amp, cmap='gray')
    plt.show()

    imshow(inv_image_pha, cmap='gray')
    plt.show()


def noise_redux_in_freq_domain(image):
    four_spec_og_img = np.fft.fftshift(np.fft.fft2(image))
    imshow(np.log(np.abs(four_spec_og_img)), cmap='gray')
    plt.show()

    noisy_image = skimage.util.random_noise(image, var=0.005)
    four_spec_noise_img = np.fft.fftshift(np.fft.fft2(noisy_image))
    imshow(np.log(np.abs(four_spec_noise_img)), cmap='gray')
    plt.show()

    image_height = image[0].size
    image_width = image_height

    radius = 20
    h = disk(radius)

    h_freq = np.zeros((image_height, image_width))
    h_freq[image_height//2 - radius : image_height//2 + radius + 1, image_width//2 - radius : image_width//2 + radius + 1] = h

    imshow(h_freq, cmap='gray')
    plt.show()

    filtered_image = np.fft.ifft2(np.fft.ifftshift(four_spec_noise_img * h_freq))
    # imshow(np.abs(filtered_image))
    plt.show()

    image_LPF_r20_PSNR = np.abs(PSNR(image, filtered_image))
    print("PSNR Absolute Value:", image_LPF_r20_PSNR)

    gaussian_lpf = gaussian_filter(image_height, image_width, 60)
    gaussian_filtered_image = np.fft.ifft2(np.fft.ifftshift(four_spec_noise_img * gaussian_lpf))

    imshow(np.abs(gaussian_filtered_image), cmap='gray')
    plt.show()

    image_gaus_LPF_PSNR = np.abs(PSNR(image, gaussian_filtered_image))
    print("PSNR Absolute Value:", image_gaus_LPF_PSNR)

    return

def filter_design(image):
    imshow(image)
    plt.show()

    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(image)))))
    plt.show()

    image_height = image[0].size
    image_width = image_height

    four_spec_img = np.fft.fftshift(np.fft.fft2(image))

    filter = np.ones((image_height, image_width))

    filter[64, 64] = 0
    filter[118, 104] = 0
    filter[138, 152] = 0
    filter[192, 192] = 0
    imshow(filter, cmap='gray')
    plt.show()

    filtered_image = np.fft.ifft2(np.fft.ifftshift(four_spec_img * filter))

    imshow(np.abs(filtered_image), cmap='gray')
    plt.show()

    return

def main():
    fourier_analysis(lena)
    noise_redux_in_freq_domain(lena)
    filter_design(frequnoisy)


if __name__ == '__main__':
    main()
