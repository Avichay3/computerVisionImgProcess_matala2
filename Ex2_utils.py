import numpy as np
from cv2 import cv2 as cv


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
    result = in_signal.size + k_size.size - 1

    """" at first, needs to flip the kernel """
    start_index = 0
    end_index = len(k_size) - 1
    while end_index > start_index:
        k_size[start_index], k_size[end_index] = k_size[end_index], k_size[start_index]
        start_index += 1
        end_index += 1






def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """




## part 2 of the assignment

def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
        """
Calculate gradient of an image
:param in_image: Gray scale image
:return: (directions, magnitude)
    """




##  part 2.2
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