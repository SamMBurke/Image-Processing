from skimage.color import rgb2gray
from skimage.io import imread, imshow
from skimage.transform import *
from skimage.exposure import *
import numpy as np
import matplotlib.pyplot as plt
plt.gray()

lena= rgb2gray(imread('lena.tiff')) * 255
cameraman = imread('cameraman.tif').astype(np.float64)
tire = imread('tire.tif').astype(np.float64) / 255.0

def PSNR(f, g) -> np.float64:
    MAXf = 255.0
    rows = len(f)  # number of rows in image f
    cols = len(f[0])  # number of columns in image f
    MSE = (1/(rows * cols)) * np.sum((f-g) ** 2)
    PSNR_out: np.float64 = 10*np.log10((MAXf ** 2)/(MSE))

    return PSNR_out


def dig_zoom(lena, cam) -> None:
    # grey scaling
    cam_gray = cam / 255.0
    lena_gray = lena / 255.0

    imshow(cam_gray)
    plt.show()

    imshow(lena_gray)
    plt.show()

    rows_l = len(lena)
    cols_l = len(lena[0])
    rows_c = len(cam)
    cols_c = len(cam[0])

    # reducing resolution by a factor of 4 using bilinear interpolation (order = 1)
    lena_down = resize(image=lena_gray, output_shape=(int(rows_l/4), int(cols_l/4)), order=1, anti_aliasing=True)
    cam_down = resize(image=cam_gray, output_shape=(int(rows_c/4), int(cols_c/4)), order=1, anti_aliasing=True)

    imshow(lena_down)
    plt.show()

    imshow(cam_down)
    plt.show()

    # increase resolution using nearest neighbour interpolation
    lena_up_NNI = resize(image=lena_down, output_shape=(int(rows_l), int(cols_l)), order=0, anti_aliasing=False)
    cam_up_NNI = resize(image=cam_down, output_shape=(int(rows_c), int(cols_c)), order=0, anti_aliasing=False)

    imshow(lena_up_NNI)
    plt.show()

    imshow(cam_up_NNI)
    plt.show()

    # increase resolution using bilinear interpolation
    lena_up_BLI = resize(image=lena_down, output_shape=(int(rows_l), int(cols_l)), order=1, anti_aliasing=False)
    cam_up_BLI = resize(image=cam_down, output_shape=(int(rows_c), int(cols_c)), order=1, anti_aliasing=False)

    imshow(lena_up_BLI)
    plt.show()

    imshow(cam_up_BLI)
    plt.show()

    # increase resolution using bicubic interpolation
    lena_up_BCI = resize(image=lena_down, output_shape=(int(rows_l), int(cols_l)), order=3, anti_aliasing=False)
    cam_up_BCI = resize(image=cam_down, output_shape=(int(rows_c), int(cols_c)), order=3, anti_aliasing=False)

    imshow(lena_up_BCI)
    plt.show()

    imshow(cam_up_BCI)
    plt.show()


def img_enhance(tire):
    # normal tire
    imshow(tire)
    plt.show()

    # tire histogram
    tire_flat = tire.flatten()
    plt.hist(tire_flat)
    plt.show()

    # tire with colours inverted
    neg_tire = 1 - tire
    imshow(neg_tire)
    plt.show()

    # colour-inverted (negative) tire histogram
    neg_tire_flat = 1 - tire_flat
    plt.hist(neg_tire_flat)
    plt.show()

    # gamma enhance by 0.5
    gamma = 0.5
    tire_enhance = tire ** gamma
    imshow(tire_enhance)
    plt.show()

    tire_enhance_flat = tire_enhance.flatten()
    plt.hist(tire_enhance_flat)
    plt.show()

    # gamma enhance by 1.3
    gamma = 1.3
    tire_enhance = tire ** gamma
    imshow(tire_enhance)
    plt.show()

    tire_enhance_flat = tire_enhance.flatten()
    plt.hist(tire_enhance_flat)
    plt.show()

    # histogram equalization
    tire_equal = equalize_hist(tire)
    imshow(tire_equal)
    plt.show()

    tire_equal_flat = tire_equal.flatten()
    plt.hist(tire_equal_flat)
    plt.show()


def main():
    # dig_zoom(lena, cameraman)
    # img_enhance(tire)
    tire_array = np.array(tire, dtype=float)
    print(tire_array.shape)
    imshow(tire_array)
    plt.show()


if __name__ == '__main__':
    main()
