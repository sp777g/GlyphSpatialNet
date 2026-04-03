import os
import shutil
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from PIL import Image, ImageFont, ImageDraw
from glob import glob


def get_coordinate():
    img_size = 128
    #################################################
    factor = 0.8
    baseline_offset = 0.1
    char_size = int(factor * img_size + 0.5)
    start_x = int(img_size / 2 - char_size / 2 + 0.5)
    start_y = int(img_size / 2 + char_size / 2 - baseline_offset * img_size + 0.5)
    #################################################

    return {'char_size': char_size, 'start_x': start_x, 'start_y': start_y}


def open_font(font_path):
    char_size = get_coordinate()['char_size']
    font = ImageFont.truetype(font_path, size=char_size)
    return font


def render(font, char):
    coord = get_coordinate()
    start_x = coord['start_x']
    start_y = coord['start_y']
    xy = (start_x, start_y)

    image = Image.new(mode="L", size=(128, 128), color=255)
    draw = ImageDraw.Draw(im=image)
    draw.text(xy=xy, text=char, font=font, anchor='ls')

    return image


def get_ttf_paths(path: str):
    ttf_path = glob(f'{path}/*.ttf') + glob(f'{path}/*.otf') + glob(f'{path}/*.ttc')

    return ttf_path


def tensor_to_pil(img, padding, n_row, func=None, img_type='L'):
    img = img * 0.5 + 0.5
    img = torchvision.utils.make_grid(tensor=img, padding=padding, nrow=n_row)
    ndarr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    img = Image.fromarray(ndarr).convert(img_type)

    if func is not None and n_row == 1:
        img = func(img)

    return img


def get_aug_params(p=1.0):
    sx = 1 if np.random.random() >= p else np.clip(np.random.randn() / 30, a_min=-0.1, a_max=0.1) + 1
    sy = 1 if np.random.random() >= p else np.clip(np.random.randn() / 30, a_min=-0.1, a_max=0.1) + 1
    theta_deg = 0 if np.random.random() >= p else np.clip(np.random.randn() * 60, a_min=-180, a_max=180)
    shx = 0 if np.random.random() >= p else np.clip(np.random.randn() / 15, a_min=-0.2, a_max=0.2)
    shy = 0 if np.random.random() >= p else np.clip(np.random.randn() / 15, a_min=-0.2, a_max=0.2)
    dx, dy = 0, 0

    return {'sx': sx, 'sy': sy, 'theta_deg': theta_deg, 'shx': shx, 'shy': shy, 'dx': dx, 'dy': dy}


def get_affine_matrix(sx, sy, theta_deg, shx, shy, dx, dy):
    theta_rad = theta_deg * np.pi / 180
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    a = sx * (cos_theta + shx * sin_theta)
    b = sy * (shx * cos_theta - sin_theta)
    c = sx * (shy * cos_theta + sin_theta)
    d = sy * (cos_theta - shy * sin_theta)

    m = torch.tensor([[a, b, dx],
                      [c, d, dy]]).float()

    return m


def aug_transform(img_size, aug=False, p=0.5, img_rev_p=0.5, input_img_sz=128):
    if not aug:
        aug_trans = [lambda img: img]
    elif np.random.random() >= p:
        aug_trans = [lambda img: img]
    else:
        aug_params = get_aug_params()
        affine_matrix = get_affine_matrix(**aug_params)
        grid = F.affine_grid(affine_matrix.unsqueeze(0), [1, 1, input_img_sz, input_img_sz], align_corners=True)

        aug_trans = [lambda img: F.grid_sample(
            img,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True)]

        # is_rev = np.random.random() < img_rev_p
        # if is_rev:
        #     aug_trans.append(lambda img: 1 - img)

    return transforms.Compose([
        transforms.ToTensor(),
        # lambda img: img.unsqueeze(0),
        # *aug_trans,
        # lambda img: F.interpolate(img, size=[img_size, img_size], mode='bilinear', align_corners=True, antialias=True),
        # lambda img: img.squeeze(0),
        transforms.Normalize([0.5], [0.5])
    ])


'''
@torch.no_grad()
def get_mass_center(img):
    """
    :param img: 一张PIL格式的灰度图像
    """

    assert isinstance(img, Image.Image) and img.mode == 'L'
    img = 1. - torchvision.transforms.ToTensor()(img)

    assert len(img.shape) == 3 and img.shape[0] == 1
    img = img.squeeze(0)
    height, width = img.shape

    row_mass = img.sum(dim=1)
    col_mass = img.sum(dim=0)

    y_indices = torch.arange(height, dtype=torch.float32, device=img.device)
    x_indices = torch.arange(width, dtype=torch.float32, device=img.device)

    centroid_y = (row_mass * y_indices).sum() / (row_mass.sum() + 1e-12)
    centroid_x = (col_mass * x_indices).sum() / (col_mass.sum() + 1e-12)

    dx = (height - 1) / 2 - centroid_x
    dy = (width - 1) / 2 - centroid_y

    return dx.item(), dy.item()


def move_to_mass_center(img):
    dx, dy = get_mass_center(img)

    transform_matrix = (1, 0, -dx, 0, 1, -dy)
    img = img.transform(
        size=img.size,
        method=Image.AFFINE,
        data=transform_matrix,
        resample=Image.BILINEAR,
        fillcolor=255
    )

    return img
'''
