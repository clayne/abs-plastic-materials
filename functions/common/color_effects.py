# Author: Christopher Gearhart

# System imports
from numba import cuda, jit, prange
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb

# Blender imports
# NONE!

# Module imports
from .images import *


@jit(nopython=True, parallel=True)
def initialize_image_texture(width, height, pixels, channels, new_channels, channel_divisor=1, flip_vertical=False):
    new_pixels = np.empty(width * height * new_channels)
    for i in prange(width * height):
        if flip_vertical:
            x = i / width
            col = round((x % 1) * width)
            row = round(x - (x % 1))
            row = height - row - 1  # here is where the vertical flip happens
            idx2 = (row * width + col) * channels
        else:
            idx2 = i * channels
        if new_channels == 3:
            idx1 = i * 3
            new_pixels[idx1 + 0] = pixels[idx2 + 0] / channel_divisor
            new_pixels[idx1 + 1] = pixels[idx2 + (1 if channels >= 3 else 0)] / channel_divisor
            new_pixels[idx1 + 2] = pixels[idx2 + (2 if channels >= 3 else 0)] / channel_divisor
        else:
            new_pixels[i] = pixels[idx2 + channels - 1] / channel_divisor
    return new_pixels


def initialize_gradient_texture(width, height, quadratic=False):
    pixels = np.empty((width, height))
    for col in prange(height):
        val = 1 - (height - 1 - col) / (height - 1)
        if quadratic:
            val = val ** 0.5
        pixels[col, :] = val
    pixels = get_1d_pixel_array(pixels)
    return pixels


def convert_channels(num_pix, channels, old_pixels, old_channels):
    old_pixels = get_2d_pixel_array(old_pixels, old_channels)
    new_pixels = np.empty((num_pix, channels))
    if channels > old_channels:
        if old_channels == 1:
            for i in range(channels):
                new_pixels[:, i] = old_pixels[:, 0]
        elif old_channels == 3:
            new_pixels[:, :3] = old_pixels[:, :3]
            new_pixels[:, 3] = 1
    elif channels < old_channels:
        if channels == 1:
            new_pixels[:, 0] = 0.2126 * old_pixels[:, 0] + 0.7152 * old_pixels[:, 1] + 0.0722 * old_pixels[:, 2]
        elif channels == 3:
            new_pxiels[:, :3] = old_pixels[:, :3]
    new_pixels = get_1d_pixel_array(new_pixels)
    return new_pixels


def set_alpha_channel(num_pix, old_pixels, old_channels, value):
    old_pixels = get_2d_pixel_array(old_pixels, old_channels)
    new_pixels = np.empty((num_pix, 4))
    new_pixels[:, :3] = old_pixels[:, :3]
    new_pixels[:, 3] = value
    new_pixels = get_1d_pixel_array(new_pixels)
    return new_pixels


@jit(nopython=True, parallel=True)
def resize_pixels(size, channels, old_pixels, old_size):
    new_pixels = np.empty(size[0] * size[1] * channels)
    for col in prange(size[0]):
        col1 = int((col / size[0]) * old_size[0])
        for row in range(size[1]):
            row1 = int((row / size[1]) * old_size[1])
            pixel_number = (size[0] * row + col) * channels
            pixel_number_ref = (old_size[0] * row1 + col1) * channels
            for ch in range(channels):
                new_pixels[pixel_number + ch] = old_pixels[pixel_number_ref + ch]
    return new_pixels


@jit(nopython=True, parallel=True)
def resize_pixels_preserve_borders(size, channels, old_pixels, old_size):
    new_pixels = np.empty(len(old_pixels))
    offset_col = int((old_size[0] - size[0]) / 2)
    offset_row = int((old_size[1] - size[1]) / 2)
    for col in prange(old_size[0]):
        col1 = int(((col - offset_col) / size[0]) * old_size[0])
        for row in range(old_size[1]):
            row1 = int(((row - offset_row) / size[1]) * old_size[1])
            pixel_number = (old_size[0] * row + col) * channels
            if 0 <= col1 < old_size[0] and 0 <= row1 < old_size[1]:
                pixel_number_ref = (old_size[0] * row1 + col1) * channels
                for ch in range(channels):
                    new_pixels[pixel_number + ch] = old_pixels[pixel_number_ref + ch]
            else:
                for ch in range(channels):
                    new_pixels[pixel_number + ch] = 0
    return new_pixels


@jit(nopython=True, parallel=True)
def crop_pixels(size, channels, old_pixels, old_size):
    new_pixels = np.empty(size[0] * size[1] * channels)
    offset_col = (old_size[0] - size[0]) // 2
    offset_row = (old_size[1] - size[1]) // 2
    for col in prange(size[0]):
        col1 = col + offset_col
        for row in range(size[1]):
            row1 = row + offset_row
            pixel_number = (size[0] * row + col) * channels
            pixel_number_ref = (old_size[0] * row1 + col1) * channels
            for ch in range(channels):
                new_pixels[pixel_number + ch] = old_pixels[pixel_number_ref + ch]
    return new_pixels


@jit(nopython=True, parallel=True)
def pad_pixels(size, channels, old_pixels, old_size):
    new_pixels = np.empty(size[0] * size[1] * channels)
    offset_col = (size[0] - old_size[0]) // 2
    offset_row = (size[1] - old_size[1]) // 2
    for col in prange(size[0]):
        col1 = col - offset_col
        for row in range(size[1]):
            row1 = row - offset_row
            pixel_number = (size[0] * row + col) * channels
            for ch in range(channels):
                if 0 <= col1 < old_size[0] and 0 <= row1 < old_size[1]:
                    pixel_number_old = (old_size[0] * row1 + col1) * channels
                    new_pixels[pixel_number + ch] = old_pixels[pixel_number_old + ch]
                else:
                    new_pixels[pixel_number + ch] = 0
    return new_pixels


def blend_pixels(im1_pixels, im2_pixels, width, height, channels, operation, use_clamp, factor_pixels):
    new_pixels = np.empty((width * height, channels))
    im1_pixels = get_2d_pixel_array(im1_pixels, channels)
    im2_pixels = get_2d_pixel_array(im2_pixels, channels)
    if isinstance(factor, np.ndarray):
        new_factor = np.empty((len(factor), channels))
        for i in range(channels):
            new_factor[:, i] = factor
        factor = new_factor
    if operation == "MIX":
        new_pixels = im1_pixels * (1 - factor) + im2_pixels * factor
    elif operation == "ADD":
        new_pixels = im1_pixels + im2_pixels * factor
    elif operation == "SUBTRACT":
        new_pixels = im1_pixels - im2_pixels * factor
    elif operation == "MULTIPLY":
        new_pixels = im1_pixels * ((1 - factor) + im2_pixels * factor)
    elif operation == "DIVIDE":
        new_pixels = im1_pixels / ((1 - factor) + im2_pixels * factor)
    elif operation == "POWER":
        new_pixels = im1_pixels ** ((1 - factor) + im2_pixels * factor)
    # elif operation == "LOGARITHM":
    #     new_pixels = math.log(im1_pixels, im2_pixels)
    elif operation == "SQUARE ROOT":
        new_pixels = np.sqrt(im1_pixels)
    elif operation == "ABSOLUTE":
        new_pixels = abs(im1_pixels)
    elif operation == "MINIMUM":
        new_pixels = np.clip(im1_pixels, a_min=im2_pixels, a_max=im1_pixels)
    elif operation == "MAXIMUM":
        new_pixels = np.clip(im1_pixels, a_min=im1_pixels, a_max=im2_pixels)
    elif operation == "LESS THAN":
        new_pixels = (im1_pixels < im2_pixels).astype(int)
    elif operation == "GREATER THAN":
        new_pixels = (im1_pixels > im2_pixels).astype(int)
    elif operation == "ROUND":
        new_pixels = np.round(im1_pixels)
    elif operation == "FLOOR":
        new_pixels = np.floor(im1_pixels)
    elif operation == "CEIL":
        new_pixels = np.ceil(im1_pixels)
    # elif operation == "FRACT":
    #     new_pixels =
    elif operation == "MODULO":
        new_pixels = im1_pixels % im2_pixels

    new_pixels = get_1d_pixel_array(new_pixels)
    if use_clamp:
        np.clip(new_pixels, 0, 1, new_pixels)

    return new_pixels


def math_operation_on_pixels(pixels, operation, clamp, value):
    new_pixels = np.empty(pixels.size)
    if operation == "ADD":
        new_pixels = pixels + value
    elif operation == "SUBTRACT":
        new_pixels = pixels - value
    elif operation == "MULTIPLY":
        new_pixels = pixels * value
    elif operation == "DIVIDE":
        new_pixels = pixels / value
    elif operation == "POWER":
        new_pixels = pixels ** value
    # elif operation == "LOGARITHM":
    #     for i in prange(new_pixels.size):
    #         new_pixels = math.log(pixels, value)
    elif operation == "SQUARE ROOT":
        new_pixels = np.sqrt(pixels)
    elif operation == "ABSOLUTE":
        new_pixels = abs(pixels)
    elif operation == "MINIMUM":
        new_pixels = np.clip(pixels, a_min=value, a_max=pixels)
    elif operation == "MAXIMUM":
        new_pixels = np.clip(pixels, a_min=pixels, a_max=value)
    elif operation == "LESS THAN":
        new_pixels = (pixels < value).astype(int)
    elif operation == "GREATER THAN":
        new_pixels = (pixels > value).astype(int)
    elif operation == "ROUND":
        new_pixels = np.round(pixels)
    elif operation == "FLOOR":
        new_pixels = np.floor(pixels)
    elif operation == "CEIL":
        new_pixels = np.ceil(pixels)
    # elif operation == "FRACT":
    #     result =
    elif operation == "MODULO":
        new_pixels = pixels % value
    elif operation == "SINE":
        result = np.sin(pixels)
    elif operation == "COSINE":
        result = np.cos(pixels)
    elif operation == "TANGENT":
        result = np.tan(pixels)
    elif operation == "ARCSINE":
        result = np.arcsin(pixels)
    elif operation == "ARCCOSINE":
        result = np.arccos(pixels)
    elif operation == "ARCTANGENT":
        result = np.arctan(pixels)
    elif operation == "ARCTAN2":
        result = np.arctan2(pixels)  #, value)

    if clamp:
        np.clip(new_pixels, 0, 1, new_pixels)

    return new_pixels


def clamp_pixels(pixels, minimum, maximum):
    return np.clip(pixels, minimum, maximum)


def adjust_bright_contrast(pixels, bright, contrast):
    return contrast * (pixels - 0.5) + 0.5 + bright


def adjust_hue_saturation_value(pixels, hue, saturation, value, channels=3):
    assert channels in (3, 4)
    pixels = get_2d_pixel_array(pixels, channels)
    hue_adjust = hue - 0.5
    pixels[:, 0] = (pixels[:, 0] + hue_adjust) % 1
    pixels[:, 1] = pixels[:, 1] * saturation
    pixels[:, 2] = pixels[:, 2] * value
    return pixels


def invert_pixels(pixels, factor, channels):
    pixels = get_2d_pixel_array(pixels, channels)
    inverted_factor = 1 - factor
    if channels == 4:
        pixels[:, :3] = (inverted_factor * pixels[:, :3]) + (factor * (1 - pixels[:, :3]))
    else:
        pixels = (inverted_factor * pixels) + (factor * (1 - pixels))
    pixels = get_1d_pixel_array(pixels)
    return pixels


@jit(nopython=True, parallel=True)
def dilate_pixels_dist(old_pixels, pixel_dist, width, height):
    mult = 1 if pixel_dist[0] > 0 else -1
    new_pixels = np.empty(len(old_pixels))
    # for i in prange(width * height):
    #     x = i / height
    #     row = round((x % 1) * height)
    #     col = round(x - (x % 1))
    for col in prange(width):
        for row in prange(height):
            pixel_number = width * row + col
            max_val = old_pixels[pixel_number]
            for c in range(-pixel_dist[0], pixel_dist[0] + 1):
                for r in range(-pixel_dist[1], pixel_dist[1] + 1):
                    if not (0 < col + c < width and 0 < row + r < height):
                        continue
                    width_amt = abs(c) / pixel_dist[0]
                    height_amt = abs(r) / pixel_dist[1]
                    ratio = (width_amt - height_amt) / 2 + 0.5
                    weighted_dist = pixel_dist[0] * ratio + ((1 - ratio) * pixel_dist[1])
                    dist = ((abs(c)**2 + abs(r)**2) ** 0.5)
                    if dist > weighted_dist + 0.5:
                        continue
                    pixel_number1 = width * (row + r) + (col + c)
                    cur_val = old_pixels[pixel_number1]
                    if cur_val * mult > max_val * mult:
                        max_val = cur_val
            new_pixels[pixel_number] = max_val
    return new_pixels


@jit(nopython=True, parallel=True)
def dilate_pixels_step(old_pixels, pixel_dist, width, height):
    mult = 1 if pixel_dist[0] > 0 else -1
    new_pixels = np.empty(len(old_pixels))
    # for i in prange(width * height):
    #     x = i / height
    #     row = round((x % 1) * height)
    #     col = round(x - (x % 1))
    for col in prange(width):
        for row in range(height):
            pixel_number = width * row + col
            max_val = old_pixels[pixel_number]
            for c in range(-pixel_dist[0], pixel_dist[0] + 1):
                if not 0 < col + c < width:
                    continue
                pixel_number1 = width * row + (col + c)
                cur_val = old_pixels[pixel_number1]
                if cur_val * mult > max_val * mult:
                    max_val = cur_val
            new_pixels[pixel_number] = max_val
    old_pixels = new_pixels
    new_pixels = np.empty(len(old_pixels))
    for col in prange(width):
        for row in range(height):
            pixel_number = width * row + col
            max_val = old_pixels[pixel_number]
            for r in range(-pixel_dist[1], pixel_dist[1] + 1):
                if not 0 < row + r < height:
                    continue
                pixel_number1 = width * (row + r) + col
                cur_val = old_pixels[pixel_number1]
                if cur_val * mult > max_val * mult:
                    max_val = cur_val
            new_pixels[pixel_number] = max_val
    return new_pixels


@jit(nopython=True, parallel=True)
def flip_pixels(old_pixels, flip_x, flip_y, width, height, channels):
    new_pixels = np.empty(len(old_pixels))
    for col in prange(width):
        col2 = int((width - col - 1) if flip_x else col)
        for row in prange(height):
            idx = (width * row + col) * channels
            row2 = int((height - row - 1) if flip_y else row)
            flipped_idx = (width * row2 + col2) * channels
            new_pixels[idx:idx + channels] = old_pixels[flipped_idx:flipped_idx + channels]
    return new_pixels


@jit(nopython=True, parallel=True)
def translate_pixels(old_pixels, translate_x, translate_y, wrap_x, wrap_y, width, height, channels):
    new_pixels = np.empty(len(old_pixels))
    for col in prange(width):
        col2 = col - translate_x
        if wrap_x:
            col2 = col2 % width
        for row in prange(height):
            row2 = row - translate_y
            if wrap_y:
                row2 = row2 % height
            idx = (width * row + col) * channels
            if not (0 <= row2 < height and 0 <= col2 < width):
                for ch in range(channels):
                    new_pixels[idx + ch] = 0
            else:
                trans_idx = round((width * row2 + col2) * channels)
                new_pixels[idx:idx + channels] = old_pixels[trans_idx:trans_idx + channels]
    return new_pixels
