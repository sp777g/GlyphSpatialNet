import glob
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import warnings
import argparse

from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings('ignore')


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

    list_path_fake = sorted(get_dir(path_fake))
    list_path_real = sorted(get_dir(path_real))

    # norm_l1
    norm_l1_list = []
    norm_l1_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                             transforms.ToTensor()])

    # norm_rmse
    norm_rmse_list = []
    norm_rmse_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                               transforms.ToTensor()])

    print('\n\n')

    for idx, path_fake_and_path_real in enumerate(zip(list_path_fake, list_path_real)):
        path_fake, path_real = path_fake_and_path_real
        assert os.path.basename(os.path.normpath(path_fake)) == os.path.basename(os.path.normpath(path_real))
        ttf_name = os.path.basename(os.path.normpath(path_fake))

        print('\n', f'[{idx + 1}|{len(list_path_fake)}]', str(os.path.join(root, ttf_name)))

        # norm_l1
        data = get_loader(path_fake, path_real, norm_l1_transforms)
        tmp_list_for_display = []
        for img_pair in data:
            fake = img_pair['fake'].to(device)
            real = img_pair['real'].to(device)

            norm_l1 = torch.abs(fake - real).mean().item() / ((1. - real).mean() + 1e-8).item()
            norm_l1_list.append(norm_l1)
            tmp_list_for_display.append(norm_l1)

        print('norm_l1:', np.mean(tmp_list_for_display).round(4))

        # norm_rmse
        data = get_loader(path_fake, path_real, norm_rmse_transforms)
        tmp_list_for_display = []
        for img_pair in data:
            fake = img_pair['fake'].to(device)
            real = img_pair['real'].to(device)

            norm_rmse = torch.sqrt(F.mse_loss(fake, real)).item() / ((1. - real).mean() + 1e-8).item()
            norm_rmse_list.append(norm_rmse)
            tmp_list_for_display.append(norm_rmse)

        print('norm_rmse:', np.mean(tmp_list_for_display).round(4))

    return {
        'norm_l1': np.mean(norm_l1_list).round(4),
        'norm_rmse': np.mean(norm_rmse_list).round(4)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    res = main(args.dir)
    print(f'\nFinal:', res)
