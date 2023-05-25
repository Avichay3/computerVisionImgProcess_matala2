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
# I have implement the second function


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






def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles, [(x, y, radius), (x, y, radius), ...]
    """
    circles_list = []
    threshold = 2
    if img.max() <= 1:  # normalize image intensity if necessary
        img = (img * 255).astype('uint8')

    max_radius = min(max_radius, min(img.shape) // 2)
    accumulator = np.zeros((len(img), len(img[0]), max_radius + 1), dtype=int)  # initialize accumulator matrix
    x_derivative = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=threshold)
    y_derivative = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=threshold)
    direction = np.degrees(np.arctan2(y_derivative, x_derivative))

    radius_step = max(1, (max_radius - min_radius) // 10)
    edges = cv.Canny(img, 75, 150)  # detect edges using Canny edge detection

    for x in range(len(edges)):  # iterate over edge pixels
        for y in range(len(edges[0])):
            if edges[x, y] == 255:
                for radius in range(min_radius, max_radius + 1, radius_step):
                    angle = direction[x, y] - 90
                    x1, y1 = int(x - radius * np.cos(np.radians(angle))), int(y + radius * np.sin(np.radians(angle)))
                    x2, y2 = int(x + radius * np.cos(np.radians(angle))), int(y - radius * np.sin(np.radians(angle)))
                    if 0 < x1 < len(accumulator) and 0 < y1 < len(accumulator[0]):
                        accumulator[x1, y1, radius] += 1
                    if 0 < x2 < len(accumulator) and 0 < y2 < len(accumulator[0]):
                        accumulator[x2, y2, radius] += 1

    threshold = np.max(accumulator) * 0.5 + 1
    x, y, radius = np.where(accumulator >= threshold)
    circles_list.extend((y[i], x[i], radius[i]) for i in range(len(x)) if x[i] != 0 or y[i] != 0 or radius[i] != 0)

    return circles_list