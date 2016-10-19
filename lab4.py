# CS4243 - Lab 4
# Name: Lau Yun Hui Barry
# Matric no.: A0111682M
# Fri 10am session

import cv2
import numpy as np
import matplotlib.pyplot as plt


def MyConvolve(img, ff):
    """ Perform convolution on image with selected filter """
    result = np.zeros(img.shape)
    x_len = img.shape[0]
    y_len = img.shape[1]

    ff = np.flipud(np.fliplr(ff))   # Flip filters

    # Apply filter to pixels
    for x in range(1, x_len - 1):
        for y in range(1, y_len - 1):
            # Left column
            top_left = img[x - 1, y - 1] * ff[0, 0]
            left = img[x, y - 1] * ff[1, 0]
            btm_left = img[x + 1, y - 1] * ff[2, 0]
            # Middle column
            top = img[x - 1, y] * ff[0, 1]
            middle = img[x, y] * ff[1, 1]
            btm = img[x + 1, y] * ff[2, 1]
            # Right column
            top_right = img[x - 1, y + 1] * ff[0, 2]
            right = img[x, y + 1] * ff[1, 2]
            btm_right = img[x + 1, y + 1] * ff[2, 2]

            result[x, y] = top_left + left + btm_left + top + middle + btm + top_right + right + btm_right

    return result


def gauss_kernels(size, sigma=1.0):
    """ Returns a 2d gaussian kernel """
    if size < 3:
        size = 3

    m = size / 2
    x, y = np.mgrid[-m:m + 1, -m:m + 1]
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel_sum = kernel.sum()

    if not sum == 0:
        kernel = kernel / kernel_sum

    return kernel


def harris_corner_detector(img, image_name):
    """ Performs harris corner detector on image """
    x_len = img.shape[0]
    y_len = img.shape[1]

    horizontal_sobel_filter = np.array([[1, 2, 1],
                                        [0, 0, 0],
                                        [-1, -2, -1]])

    vertical_sobel_filter = np.array([[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]])

    # Compute edge strength of pixels in image
    gx = MyConvolve(img, horizontal_sobel_filter) 
    gy = MyConvolve(img, vertical_sobel_filter)

    # Compute product of derivatives
    I_xx = gx * gx
    I_xy = gx * gy
    I_yy = gy * gy

    # Define Gaussian kernel
    kernel = gauss_kernels(3)
    
    # Convolve product of derivatives (I_xx, I_xy, I_yy)
    W_xx = MyConvolve(I_xx, kernel)
    W_xy = MyConvolve(I_xy, kernel)
    W_yy = MyConvolve(I_yy, kernel)

    # Initialise response matrix
    response = np.zeros(img.shape)

    # Compute response for pixels that will be taken into consideration
    for x in range(1, x_len - 1):
        for y in range(1, y_len - 1):
            w = np.array([[W_xx[x, y], W_xy[x, y]],
                          [W_xy[x, y], W_yy[x, y]]])
            det_W = np.linalg.det(w)
            trace_W = np.trace(w)
            response[x, y] = det_W - 0.06 * trace_W * trace_W

    # Get max response from response matrix        
    max_r = np.max(response)        

    # Initialise lists for x, y coordinates
    x_list = []
    y_list = []

    # Select response values within 10% of maximum response
    for x in range(1, x_len - 1):
        for y in range(1, y_len - 1):
            if response[x, y] >= (max_r * 0.1) and response[x, y] <= (max_r * 1.9):
                x_list.append(x)
                y_list.append(y)

    # Plot selected response values on image
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.scatter(y_list, x_list, edgecolors='blue', facecolors='none', s=81, marker='s')
    plt.savefig(image_name + '_corners.jpg')


def main():
    image_list = ['checker', 'flower', 'test1', 'test2', 'test3']

    for image_name in image_list:
        img = cv2.imread(image_name + '.jpg', 0)
        img = harris_corner_detector(img, image_name)


# Execute main function
main()
