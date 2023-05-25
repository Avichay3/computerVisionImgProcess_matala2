import numpy as np
from cv2 import cv2 as cv
import math


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 211780267


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    # need to return the convolved array so needs to know what is the size of the convolved array
    result_size = in_signal.size + k_size.size - 1
    result = np.zeros(result_size)
    flipped_kernel = np.flip(k_size)  # Flipping the kernel using numpy function
    for i in range():  # The convolution
        for j in range(k_size.size):
            if i - j < 0 or i - j >= in_signal.size:
                continue
            result[i] = result[i] + in_signal[i - j] * flipped_kernel[j]
    return result






def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    kernel = np.flip(kernel)  # Flip the kernel horizontally and vertically
    kernel_height, kernel_width = kernel.shape
    img_height, img_width = in_image.shape
    ans_matrix = np.zeros_like(in_image)
    _pad_height = math.floor(kernel_height / 2)  # Padding for height
    _pad_width = math.floor(kernel_width / 2)  # Padding for width
    #  I choose the 'edge' mode for padding
    mat = np.pad(in_image, pad_width=((_pad_height, _pad_height), (_pad_width, _pad_width)),
                 mode='edge')

    for rows in range(img_height):
        for columns in range(img_width):
            curr_mat = mat[rows:rows + kernel_height, columns:columns + kernel_width]
            new_val = np.sum(curr_mat * kernel).round()
            ans_matrix[rows, columns] = new_val
    return ans_matrix




## part 2 of the assignment

def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
        """
    Calculate gradient of an image
    :param in_image: Gray scale image
    :return: (directions, magnitude)
        """
        kernel_x = np.array([[1, 0, -1]])
        x_derivative = cv.filter2D(in_image, -1, kernel_x, borderType = cv.BORDER_REPLICATE)
        #  compute the Y derivative using convolution with [1, 0, -1]T
        kernel_y = kernel_x.T
        y_derivative = cv.filter2D(in_image, -1, kernel_y, borderType = cv.BORDER_REPLICATE)
        #  compute the magnitude and direction using the x and y derivatives
        magnitude = np.sqrt(x_derivative ** 2 + y_derivative ** 2).astype(np.float64)
        direction = np.arctan2(y_derivative, x_derivative).astype(np.float64)
        return magnitude, direction




##  Part 2.2, this is a bonus
def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """




def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using Open CV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """




##  Part 3 Edge detection
def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """



def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    crossing_zero_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            try:
                # check if there is a "zero crossing"
                # count the positive and negative values among the neighboring pixels
                positive_count = 0
                negative_count = 0

                neighbor_pixels = [img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1], img[i, j - 1],
                                    img[i, j + 1], img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1]]

                max_pixel_value = max(neighbor_pixels)
                min_pixel_value = min(neighbor_pixels)
                for pixel in neighbor_pixels:
                    if pixel > 0:
                        positive_count += 1
                    elif pixel < 0:
                        negative_count += 1
                if negative_count > 0 and positive_count > 0: # Means that there is a zero crossing
                    if img[i, j] > 0:
                        crossing_zero_img[i, j] = img[i, j] + abs(min_pixel_value)
                    elif img[i, j] < 0:
                        crossing_zero_img[i, j] = abs(img[i, j]) + max_pixel_value
            except IndexError: # ignore pixels at the image boundaries
                pass
    return crossing_zero_img