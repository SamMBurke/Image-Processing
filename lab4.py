import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
import skimage.util
import scipy.ndimage as ndimage
from skimage.color import rgb2gray
from skimage.morphology import disk
from skimage.io import imread, imshow
import matplotlib
import skimage.restoration as restoration

import skimage.filters.rank as rank
plt.gray()
def PSNR(f,g):
    return 10*np.log10(1.0/ np.mean(np.square(f-g)))


class Selector:
    def __init__(self, ax):
        self.RS = RectangleSelector(ax, self.line_select_callback,
                                     useblit=True,
                                       button=[1, 3],  
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
        self.bbox = [None, None, None, None]
        
    def line_select_callback(self,eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.bbox = [int(y1), int(y2), int(x1), int(x2)]
    def get_bbox(self):
        return self.bbox

f = imread('cameraman.tif').astype(np.float64)/255
I = imread('degraded.tif').astype(np.float64)/255


def disk_blur(radius, image):
    h_d = disk(radius)
    h = np.zeros((256,256))

    h[0:9, 0:9] = h_d / np.sum(h_d)
    h = np.roll(h, (-5, -5))

    h_freq = np.fft.fft2(h)
    f_blurfreq = h_freq * np.fft.fft2(image)
    f_blur = np.real(np.fft.ifft2(f_blurfreq))

    return f_blur, h_freq, h


def inverse_filter(image, h_freq):
    return np.fft.ifft2((np.fft.fft2(image)) / h_freq)
    

def image_restoration_in_the_frequency_domain(image):
    imshow(image)
    plt.show()

    blurred_image, h_freq, h = disk_blur(4, image)

    # disk blur filter in frequency domain
    imshow(np.real(h_freq), cmap='gray')
    plt.show()

    # blurred image
    imshow(blurred_image, cmap='gray')
    print("blurred image PSNR:", PSNR(image, blurred_image))
    plt.show()

    # blurred image inverse filtered
    inverse_filtered_image = inverse_filter(blurred_image, h_freq)
    imshow(np.real(inverse_filtered_image), cmap='gray')
    print("blurred image inverse filtered PSNR:", PSNR(image, np.real(inverse_filtered_image)))
    plt.show()

    # blurred image with gaussian noise
    gaussian_noise_blurred_image = skimage.util.random_noise(blurred_image, var=0.002)
    imshow(gaussian_noise_blurred_image, cmap='gray')
    print("blurred image with gaussian noise PSNR:", PSNR(image, gaussian_noise_blurred_image))
    plt.show()

    # blurred image with gaussian noise inverse filtered
    inverse_filtered_gaussian_image = inverse_filter(gaussian_noise_blurred_image, h_freq)
    imshow(np.real(inverse_filtered_gaussian_image), cmap='gray')
    print("blurred image with gaussian noise inverse filtered PSNR:", PSNR(image, np.real(inverse_filtered_gaussian_image)))
    plt.show()

    # blurred image with gaussian noise wiener filtered
    weiner_filtered_image = restoration.wiener(image=gaussian_noise_blurred_image, psf=np.fft.fftshift(h), balance=0.01)
    imshow(weiner_filtered_image, cmap='gray')
    print("blurred image with gaussian noise wiener filtered PSNR (balance 0.01):", PSNR(image, np.real(weiner_filtered_image)))
    plt.show()

    weiner_filtered_image = restoration.wiener(image=gaussian_noise_blurred_image, psf=np.fft.fftshift(h), balance=3.00)
    imshow(weiner_filtered_image, cmap='gray')
    print("blurred image with gaussian noise wiener filtered PSNR (balance 3.00):", PSNR(image, np.real(weiner_filtered_image)))
    plt.show()

    weiner_filtered_image = restoration.wiener(image=gaussian_noise_blurred_image, psf=np.fft.fftshift(h), balance=10.00)
    imshow(weiner_filtered_image, cmap='gray')
    print("blurred image with gaussian noise wiener filtered PSNR (balance 10.00):", PSNR(image, np.real(weiner_filtered_image)))
    plt.show()

    return


def adaptive_filtering(image):
    imshow(image)
    plt.show()

    ax = plt.gca()
    ax.imshow(image)

    select = Selector(ax)
    plt.show()

    noise_var = np.var(image[select.bbox[0]:select.bbox[1], select.bbox[2]:select.bbox[3]])

    mn = np.ones((5,5)) / 25
    local_mean = signal.convolve(image, mn, mode='same')
    local_var = signal.convolve(image**2, mn, mode='same') - local_mean**2

    K = (local_var - noise_var) / local_var

    lee_filtered_image = K * image + (1 - K) * local_mean
    imshow(lee_filtered_image, cmap='gray')
    plt.show()

    print("PSNR Lee Filter: ", PSNR(image, lee_filtered_image))
    return


def main():
    image_restoration_in_the_frequency_domain(f)
    # adaptive_filtering(I)


if __name__ == '__main__':
    main()
