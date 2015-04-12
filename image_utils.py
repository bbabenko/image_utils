import cv2
import numpy as np
from collections import namedtuple

def gblur(image, sigma):
    """
    Convenience wrapper around cv2.GaussianBlur.
    
    Parameters
    ----------
    image : 2D or 3D numpy array
    sigma : parameter for gaussian kernel
    
    Returns
    -------
    blurred : 2D or 3D array

    """
    return cv2.GaussianBlur(image, (0,0), sigma)

# convinience wrapper around color space enums in cv2 (essentially a dedicated namespace for them)
_COLORSPACE_NAME_TO_CV2_ENUM = {
    'luv': cv2.COLOR_RGB2LUV,
    'hsv': cv2.COLOR_RGB2HSV,
    'lab': cv2.COLOR_RGB2LAB,
    'gray': cv2.COLOR_RGB2GRAY,
    'grey': cv2.COLOR_RGB2GRAY,
}
COLORSPACE = namedtuple(
        'COLORSPACE', _COLORSPACE_NAME_TO_CV2_ENUM.keys())(*_COLORSPACE_NAME_TO_CV2_ENUM.values())

def rgb_convert(image, colorspace=COLORSPACE.luv):
    """
    Convert RGB image to another colorspace.  Convenience wrapper around cv2.cvtColor.

    Parameters
    ----------
    image : 3D numpy array
    colorspace : one of cv2.COLOR_RGB* enums, see COLORSPACE as a convinience wrapper for that

    Returns
    -------
    image : 2D or 3D numpy array

    """
    return cv2.cvtColor(image, colorspace)

# convinience wrapper around border style enums in cv2 (essentially a dedicated namespace for them)
_PAD_STYLE_TO_CV2_ENUM = {
    'reflect': cv2.BORDER_REFLECT_101, # not sure why you'd ever want cv2.BORDER_REFLECT
    'replicate': cv2.BORDER_REPLICATE,
    'constant': cv2.BORDER_CONSTANT,
}
PAD_STYLE = namedtuple(
        'PAD_STYLE', _PAD_STYLE_TO_CV2_ENUM.keys())(*_PAD_STYLE_TO_CV2_ENUM.values())

def pad(image, 
        (pad_left, pad_right), 
        (pad_top, pad_bottom), 
        pad_style=PAD_STYLE.reflect, 
        pad_constant=0):
    """
    Convenience wrapper around `cv2.copyMakeBorder`.

    Parameters
    ----------
    image: 2D or 3D numpy array
    (pad_left, pad_right): horizontal pad amounts, will be cast to ints
    (pad_top, pad_bottom): vertical pad amounts, will be cast to ints
    pad_style: one of cv2.BORDER* enums, see PAD_STYLE as a convinience wrapper for that
    pad_constant: if pad style is `constant`, fill with this value

    Returns
    -------
    """
    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    padded_image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right, pad_style, value=pad_constant)
    assert padded_image.shape[:2] == (image.shape[0] + pad_top + pad_bottom,
                                      image.shape[1] + pad_left + pad_right)
    assert padded_image.ndim == image.ndim
    return padded_image

def crop(image, (left_x, top_y, width, height), pad_style=PAD_STYLE.reflect, pad_constant=0):
    """
    Crop a patch from an image given a bounding box.  The bounding box is allowed to go outside the
    bounds of the image, in which case the image will first be padded with the specified padding
    style.

    Parameters
    ----------
    image : 2D or 3D numpy array
    (left_x, top_y, width, height) : bounding box parameters, will be cast to ints
    pad_style: one of cv2.BORDER* enums, see PAD_STYLE as a convinience wrapper for that
    pad_constant: if pad style is `constant`, fill with this value
    
    Returns
    -------
    patch : 2D or 3D numpy array with specified width and height

    """
    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    left_x, top_y, width, height = int(left_x), int(top_y), int(width), int(height)

    if left_x + width > image.shape[1]:
        pad_right = left_x + width - image.shape[1]
    if top_y + height > image.shape[0]:
        pad_bottom = top_y + height - image.shape[0]
    if left_x < 0:
        pad_left = -left_x
        left_x = 0
    if top_y < 0:
        pad_top = -top_y
        top_y = 0

    if pad_left or pad_right or pad_top or pad_bottom:
        image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right, pad_style, value=pad_constant)
    cropped_image = image[top_y:top_y+height, left_x:left_x+width,...]
    assert cropped_image.shape[:2] == (width, height)
    assert cropped_image.ndim == image.ndim
    return cropped_image

def gradient(image):
    """
    Compute gradient magnitude and orientation of an image.  Color images will first be turned into
    grayscale.  Input will be scaled to be between [0,1] before computing gradient.

    Parameters
    ----------
    image : 2D or 3D image
    
    Returns
    -------
    gradient_magnitude: 2D float32 numpy array, same width and height as input
    gradient_orientation: 2D float32 numpy array, same width and height as input

    """
    if image.ndim == 3:
        image = rgb_convert(image, COLORSPACE.gray)
    assert image.ndim == 2, 'Invalid number of channels'
    image = image.astype(np.float32)
    image = image / image.max()
    dy, dx = np.gradient(image)
    magnitude = np.sqrt(dx*dx + dy*dy).astype(np.float32)
    orientation = np.arctan2(dy, dx).astype(np.float32)
    return magnitude, orientation

def resize(image, width, height, keep_aspect_ratio=True):
    """
    
    Parameters
    ----------
    image : 2D or 3D image
    width : int, new width
    height : int, new height
    keep_aspect_ratio : if True, will resize image to largest possible size that fits inside
        a width x height rectangle, otherwise will distort image
    
    Returns
    -------
    """
    current_height, current_width = image.shape[:2]
    width, height = int(width), int(height)
    current_aspect_ratio = float(current_width)/current_height
    aspect_ratio = float(width)/height
    if keep_aspect_ratio:
        if current_aspect_ratio > aspect_ratio:
            height = int(width / current_aspect_ratio)
        else:
            width = int(height * current_aspect_ratio)
    resized_image = cv2.resize(image, (width, height))
    assert resized_image.shape[0] <= height and resized_image.shape[1] <= width
    if keep_aspect_ratio:
        assert np.abs(float(resized_image.shape[1])/resized_image.shape[0] - 
                      current_aspect_ratio) < .1
    assert resized_image.shape[:2] == (height, width)
    assert resized_image.ndim == image.ndim
    return resized_image
