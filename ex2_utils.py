import numpy as np
import cv2
import math


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return int(211780267)


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
    for i in range(result_size):  # The convolution
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
    kernel_flipped = np.flip(kernel)
    edge_padding_size  = kernel_flipped.shape[0] // 2 #  number of pixels to add for each axis
    img_mat = np.pad(in_image, pad_width=edge_padding_size , mode='edge')  #  padding
    result_mat = np.zeros_like(in_image)  # for store the result
    for i in range(result_mat.shape[0]):  # the convolution
        for j in range(result_mat.shape[1]):
            convolved_pixel = 0
            for k in range(kernel_flipped.shape[0]):
                for l in range(kernel_flipped.shape[1]):
                    convolved_pixel += img_mat[k + i, l + j] * kernel_flipped[k, l]
            result_mat[i, j] = convolved_pixel
    return result_mat




## part 2 of the assignment

def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
        """
    Calculate gradient of an image
    :param in_image: Gray scale image
    :return: (directions, magnitude)
        """
        kernel_x = np.array([[1, 0, -1]])
        x_derivative = cv2.filter2D(in_image, -1, kernel_x, borderType = cv2.BORDER_REPLICATE)
        #  compute the Y derivative using convolution with [1, 0, -1]T
        kernel_y = kernel_x.T
        y_derivative = cv2.filter2D(in_image, -1, kernel_y, borderType = cv2.BORDER_REPLICATE)
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
    x = np.linspace(-(k_size - 1) / 2, (k_size - 1) / 2, k_size)
    gaussian_kernel = np.exp(-0.5 * (x ** 2))
    gaussian_kernel /= np.sum(gaussian_kernel)  # normalize
    blurred_image = cv2.filter2D(in_image, -1, gaussian_kernel, borderType=cv2.BORDER_REPLICATE)  # apply gaussian kernel
    return blurred_image





def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    if kernel_size % 2 == 0:
        print("This kernel size is not an odd number!")
    kernel = cv2.getGaussianKernel(int(kernel_size), -1)
    kernel = kernel @ kernel.T
    blurred_image = cv2.filter2D(in_image, -1, kernel)
    return blurred_image




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




## part 4 for the assignment

def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    if img.max() <= 1: # normalize the image if the intensities are between 0-1
        img = (img * 255).astype('uint8')
    space = np.zeros((img.shape[0], img.shape[1], max_radius + 1))  # initialize the space array
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    direction = np.arctan2(gradient_y, gradient_x)  # the angle
    edges = cv2.Canny(img, threshold1=75, threshold2=150)  # detect edges

    for i in range(edges.shape[0]):  # iterate over the edges
        for j in range(edges.shape[1]):
            if edges[i, j] == 255:  # If this pixel is an edge
                for radius in range(min_radius, max_radius + 1):
                    angle = direction[i, j] - np.pi / 2
                    x1, y1 = int(i - radius * np.cos(angle)), int(j + radius * np.sin(angle))
                    x2, y2 = int(i + radius * np.cos(angle)), int(j - radius * np.sin(angle))
                    if 0 <= x1 < img.shape[0] and 0 <= y1 < img.shape[1]:
                        space[x1, y1, radius] += 1
                    if 0 <= x2 < img.shape[0] and 0 <= y2 < img.shape[1]:
                        space[x2, y2, radius] += 1
    threshold = np.max(space) * 0.5 + 1  # update variable
    x, y, rad = np.where(space >= threshold)
    circles_list = [(y[i], x[i], rad[i]) for i in range(len(x)) if x[i] != 0 or y[i] != 0 or rad[i] != 0]

    return circles_list






## part 5 for the assignment

def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: img image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: Opencv2 implementation, my implementation
    """
    opencv_func = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    num = (k_size - 1) // 2  # number of pixels to pad on each side of the image
    image_padded = np.pad(in_image, pad_width=num, mode='edge')
    bilateral_image = np.zeros_like(in_image)  # empty array for the filtered image
    x, y = np.indices((k_size, k_size)) - k_size // 2
    distance_diff = np.exp(-(x ** 2 + y ** 2)/(2 * sigma_space ** 2))
    for i in range(bilateral_image.shape[0]):
        for j in range(bilateral_image.shape[1]):
            center = image_padded[i:i + k_size, j:j + k_size]  # current pixel value and its neighbors
            color_diff = np.exp(- (center - in_image[i, j]) ** 2 / (2 * sigma_color ** 2))
            weight = color_diff * distance_diff  # bilateral filter weight matrix
            weight = weight / np.sum(weight)  # normalize
            bilateral_image[i, j] = np.sum(center * weight)
    return opencv_func, bilateral_image