import glob
import os
import lpips
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import warnings
import argparse

from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure

warnings.filterwarnings('ignore')


def tensor_to_pil(img, padding, n_row, func=None, img_type='L'):
    img = img * 0.5 + 0.5
    img = torchvision.utils.make_grid(tensor=img, padding=padding, nrow=n_row)
    ndarr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    img = Image.fromarray(ndarr).convert(img_type)

    if func is not None and n_row == 1:
        img = func(img)

    return img


def psnr(image_a, image_b, max_value=1.0):
    """
    计算两幅图像的PSNR值。

    :param image_a: 输入图像A，形状为(N, C, H, W)，范围在[0, max_value]
    :param image_b: 输入图像B，形状为(N, C, H, W)，范围在[0, max_value]
    :param max_value: 图像的最大可能像素值，默认为1.0（对于归一化到[0, 1]的图像）
    :return: PSNR值
    """
    mse = torch.mean((image_a.float() - image_b.float()) ** 2)
    if mse == 0:
        return float('inf')
    psnr_val = 20 * torch.log10(torch.tensor(max_value)) - 10 * torch.log10(mse)
    return psnr_val.item()


class DatasetRemap(Dataset):
    def __init__(self, path_fake, path_real, transforms, image_type):
        self.path_fake = np.sort(np.array(glob.glob(str(path_fake) + '/*.*')))
        self.path_real = np.sort(np.array(glob.glob(str(path_real) + '/*.*')))
        self.transforms = transforms
        self.image_type = image_type

    def __getitem__(self, index):
        fake_img = Image.open(str(self.path_fake[index]))
        real_img = Image.open(str(self.path_real[index]))

        fake = self.transforms(fake_img.convert(self.image_type))
        real = self.transforms(real_img.convert(self.image_type))

        return {'fake': fake, 'real': real}

    def __len__(self):
        return len(self.path_fake)


def get_loader(path_fake, path_real, transforms, image_type='RGB'):
    image_dataset = DatasetRemap(path_fake, path_real, transforms, image_type)
    dataloader = torch.utils.data.DataLoader(dataset=image_dataset, batch_size=1, shuffle=False)
    return dataloader


def get_dir(path):
    p = Path(path)
    subdirectories = [x for x in p.iterdir() if x.is_dir()]
    return subdirectories


@torch.no_grad()
def main(root):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_fake = os.path.join('./', root, 'genimgs')
    path_real = os.path.join('./', root, 'gtimgs')

    path_diff = os.path.join('./', root, 'diff')

    list_path_fake = sorted(get_dir(path_fake))
    list_path_real = sorted(get_dir(path_real))

    # LPIPS
    lpips_list = []
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    lpips_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # RMSE
    rmse_list = []
    rmse_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()])

    # SSIM
    ssim_list = []
    ssim_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()])

    # PSNR
    psnr_list = []
    psnr_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()])

    # Display diff
    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()])

    print('\n\n')

    for idx, path_fake_and_path_real in enumerate(zip(list_path_fake, list_path_real)):
        path_fake, path_real = path_fake_and_path_real
        assert os.path.basename(os.path.normpath(path_fake)) == os.path.basename(os.path.normpath(path_real))
        ttf_name = os.path.basename(os.path.normpath(path_fake))

        print('\n', f'[{idx + 1}|{len(list_path_fake)}]', str(os.path.join(root, ttf_name)))

        # # Display diff
        # data = get_loader(path_fake, path_real, data_transforms, 'L')
        # for idx, img_pair in enumerate(data):
        #     fake = img_pair['fake'].to(device)
        #     real = img_pair['real'].to(device)
        #
        #     res_p = torch.nn.functional.relu(real - fake)
        #     res_m = torch.nn.functional.relu(fake - real)
        #
        #     res_p = torch.cat([torch.zeros_like(res_p), res_p, res_p], dim=1)
        #     res_m = torch.cat([res_m, torch.zeros_like(res_m), res_m], dim=1)
        #
        #     res = 1. - (res_p + res_m)
        #
        #     fake = fake.repeat(1, 3, 1, 1)
        #     real = real.repeat(1, 3, 1, 1)
        #
        #     img = torch.cat([fake, res, real], dim=0) * 2 - 1
        #     img = tensor_to_pil(img, padding=0, n_row=3, img_type='RGB')
        #
        #     file_name = f'{idx:4d}.png'
        #     if not os.path.isdir(os.path.join(path_diff, ttf_name)):
        #         os.makedirs(os.path.join(path_diff, ttf_name))
        #
        #     img.save(os.path.join(path_diff, ttf_name, file_name))

        # LPIPS
        lpips_data = get_loader(path_fake, path_real, lpips_transforms)
        tmp_list_for_display = []
        for img_pair in lpips_data:
            fake = img_pair['fake'].to(device)
            real = img_pair['real'].to(device)

            distance = lpips_model(fake, real).item()
            lpips_list.append(distance)
            tmp_list_for_display.append(distance)

        print('LPIPS:', np.mean(tmp_list_for_display).round(4))

        # RMSE
        rmse_data = get_loader(path_fake, path_real, rmse_transforms)
        tmp_list_for_display = []
        for img_pair in rmse_data:
            fake = img_pair['fake'].to(device)
            real = img_pair['real'].to(device)

            rmse = torch.sqrt(F.mse_loss(fake, real)).item()
            rmse_list.append(rmse)
            tmp_list_for_display.append(rmse)

        print('RMSE:', np.mean(tmp_list_for_display).round(4))

        # SSIM
        ssim_data = get_loader(path_fake, path_real, ssim_transforms)
        tmp_list_for_display = []
        for img_pair in ssim_data:
            fake = img_pair['fake'].to(device)
            real = img_pair['real'].to(device)

            ssim_score = structural_similarity_index_measure(fake, real).item()
            ssim_list.append(ssim_score)
            tmp_list_for_display.append(ssim_score)

        print('SSIM:', np.mean(tmp_list_for_display).round(4))

        # PSNR
        psnr_data = get_loader(path_fake, path_real, psnr_transforms)
        tmp_list_for_display = []
        for img_pair in psnr_data:
            fake = img_pair['fake'].to(device)
            real = img_pair['real'].to(device)

            psnr_score = psnr(fake, real)
            psnr_list.append(psnr_score)
            tmp_list_for_display.append(psnr_score)

        print('PSNR:', np.mean(tmp_list_for_display).round(4))

    return {
        'lpips': np.mean(lpips_list).round(4),
        'rmse': np.mean(rmse_list).round(4),
        'ssim': np.mean(ssim_list).round(4),
        'psnr': np.mean(psnr_list).round(4)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    res = main(args.dir)
    print(f'\nFinal:', res)
