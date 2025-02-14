from skimage.color import rgb2ycbcr, ycbcr2rgb, rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage.transform import *
import numpy as np
from sklearn.cluster import KMeans

from scipy.fftpack import dct
from skimage.metrics import peak_signal_noise_ratio as PSNR

# from python_lab5_networks import networks


def sub2ind(n_row, row, col):
  return n_row * col + row


def dctmtx(N):
  return dct(np.eye(N), norm='ortho', axis=0)


def func(x, mat):
  return mat @ x @ mat.T


def func1(x, mat):
  return np.multiply(mat, x)


def blockproc(im, mat, block_sz, func):
  h, w = im.shape
  m, n = block_sz
  im_out = np.zeros_like(im)
  for x in range(0, h, m):
    for y in range(0, w, n):
      block = im[x:x+m, y:y+n]
      im_out[x:x+m, y:y+n] = func(block, mat)
  return im_out


def chroma_subsampling(im):
  imshow(im)
  plt.show()
  
  im_ycbcr = rgb2ycbcr(im)

  for i in range(3):
    imshow(im_ycbcr[:,:,i])
    plt.show()

  reduced_im_chroma = im_ycbcr
  reduced_im_luma = im_ycbcr

  # bilinear transformation for resolution reduction and subsequent upscaling of the Cb and Cr channels
  reduced_cb = resize(image=im_ycbcr[:,:,1], output_shape=(int(len(im_ycbcr)/2), int(len(im_ycbcr[0])/2)), order=1, anti_aliasing=True)
  upsampled_cb = resize(image=reduced_cb, output_shape=(int(len(im_ycbcr)), int(len(im_ycbcr[0]))), order=1, anti_aliasing=True)

  reduced_cr = resize(image=im_ycbcr[:,:,2], output_shape=(int(len(im_ycbcr)/2), int(len(im_ycbcr[0])/2)), order=1, anti_aliasing=True)
  upsampled_cr = resize(image=reduced_cr, output_shape=(int(len(im_ycbcr)), int(len(im_ycbcr[0]))), order=1, anti_aliasing=True)

  reduced_im_chroma[:,:,1] = upsampled_cb
  reduced_im_chroma[:,:,2] = upsampled_cr

  imshow(ycbcr2rgb(reduced_im_chroma))
  plt.show()

  reduced_y = resize(image=im_ycbcr[:,:,0], output_shape=(int(len(im_ycbcr)/2), int(len(im_ycbcr[0])/2)), order=1, anti_aliasing=True)
  upsampled_y = resize(image=reduced_y, output_shape=(int(len(im_ycbcr)), int(len(im_ycbcr[0]))), order=1, anti_aliasing=True)

  reduced_im_luma[:,:,0] = upsampled_y

  imshow(ycbcr2rgb(reduced_im_luma))
  plt.show()

  return


def colour_segmentation(im):
  imshow(im)
  plt.title("Original Image")
  plt.show()

  im_Lab = rgb2lab(im)

  for i in range(3):
    imshow(im_Lab[:,:,i])
    plt.title("L*a*b* Transformed Image")
    plt.show()
  
  Ks = [2, 4]
  for i in range(len(Ks)):
    im_Lab = rgb2lab(im)

    K = Ks[i]                               # else K ==4
    row = np.array([55, 200]) - 1   if K == 2 else np.array([55, 130, 200, 280]) - 1
    col = np.array([155, 400]) - 1  if K == 2 else np.array([155, 110, 400, 470]) - 1

    mu = im_Lab[row,col]
    m, n, ch = im_Lab.shape
    im_Lab = np.reshape(im_Lab, (m * n, ch), order="F")

    k_means = KMeans(n_clusters=K, init=mu).fit(im_Lab)
    cluster_idx = k_means.predict(im_Lab)

    # label each pixel according to k-means
    pixel_labels = np.reshape(cluster_idx, (m, n), order="F")
    plt.imshow(pixel_labels, cmap="jet")
    plt.title(f"K-Means Image Labeled by Cluster Index $K={K}$")
    plt.show()

    if K == 4:
      for i in range(K):
        # reinitialize the Lab transformed image 
        im_Lab = rgb2lab(im)

        # initialize array of zeros by the same shape as the image
        im_segment = np.zeros(shape=im_Lab.shape)

        # all pixels characterized by pixel _label from K Means of the image are isolated into im_segment
        im_segment[pixel_labels == i] = im_Lab[pixel_labels == i]
        
        # RGB K-Means
        plt.imshow(lab2rgb(im_segment), cmap="jet")
        plt.title(f"K-Means Image by Cluster Index ${i}$")
        plt.show()

  return


def image_transform(f):
  f = rgb2gray(f)
  imshow(f)
  plt.show()

  T = dctmtx(8)
  # imshow(T)
  # plt.show()

  F_trans = np.floor(blockproc(f - 128, T, [8, 8], func))
  imshow(np.abs(F_trans[80 : 80 + 8, 296 : 296 + 8]))
  plt.show()

  imshow(F_trans[0 : 8, 0 : 8])  
  plt.show()

  mask = np.zeros((8, 8))
  mask[0, 0] = 1
  mask[0, 1] = 1
  mask[0, 2] = 1
  mask[1, 0] = 1
  mask[1, 1] = 1
  mask[2, 0] = 1
  F_thresh = blockproc(F_trans, mask, [8, 8], func1)

  f_thresh = np.floor(blockproc(F_thresh, T.T, [8, 8], func)) + 128

  psnr = PSNR(f, f_thresh)
  imshow(f_thresh)
  plt.title(f"PSNR: {psnr}")
  plt.show()

  return


def quantization(im):
  im = rgb2gray(im) * 255
  scalers = [1, 3, 5, 10]

  for scaler in scalers:
    Z = scaler * np.array([
      [16, 11, 10, 16, 24, 40, 51, 61],
      [12, 12, 14, 19, 26, 58, 60, 55],
      [14, 13, 16, 24, 40, 57, 69, 56],
      [14, 17, 22, 29, 51, 87, 80, 62],
      [18, 22, 37, 56, 68, 109, 103, 77],
      [24, 35, 55, 64, 81, 104, 113, 92],
      [49, 64, 78, 87, 103, 121, 120, 101],
      [72, 92, 95, 98, 112, 100, 103, 99] 
    ])

    T = dctmtx(8)

    # 8x8 CDT Transform of image into subimages
    IM_DCT_trans = np.floor(blockproc(im - 128, T, [8, 8], func))

    # quantize each subimage by using the same functiont as before to apply the quantization across all subimages
    # NOTE: This makes the image compression lossy
    IM_quantized_DCT = np.floor(blockproc(IM_DCT_trans, 1 / Z, [8, 8], func1))

    # reconstruction by multiplying subimages by Z
    IM_reconstruct_DCT = blockproc(IM_quantized_DCT, Z, [8, 8], func1)

    # apply the inverse DCT transform, adding 128 back as well
    im_reconstruct = np.floor(blockproc(IM_reconstruct_DCT, T.T, [8, 8], func)) + 128

    imshow(im_reconstruct, cmap='gray')
    plt.suptitle(f"Lossy Compression Reconstruction from DCT and Quantization\nScaler:{scaler}    PSNR:{PSNR(im / 255, im_reconstruct / 255)})")
    plt.tight_layout()
    plt.show()

  return


def convolution_neural_networks(im):
  return


def main():
  # chroma_subsampling(peppers)
  # colour_segmentation(peppers)
  # image_transform(lena)
  # quantization(lena)
  return

if __name__ == "__main__":

  lena = imread('lena2.tiff').astype(np.float64) / 255
  peppers = imread('peppers.png').astype(np.float64) / 255

  main()
