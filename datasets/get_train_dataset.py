import json
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from pathlib import Path

try:
    from datasets.utils import aug_transform, get_ttf_paths, open_font, render
except ImportError:
    from FFG.datasets.utils import aug_transform, get_ttf_paths, open_font, render


class TrainDataset(Dataset):
    def __init__(self, dataset_path, img_size, num_ref_chars):
        self.source_path = os.path.join(dataset_path, 'SourceHanSerifSC-Regular.otf')
        self.source = open_font(self.source_path)

        self.font_path_list = get_ttf_paths(os.path.join(dataset_path, 'seen_fonts_for_train'))
        self.name_font_dict = {Path(path).stem: open_font(path) for path in self.font_path_list}
        self.font_name_list = list(self.name_font_dict.keys())

        self.chars_list = json.load(open(os.path.join(dataset_path, 'seen_chars_for_train.json'), encoding='utf-8'))

        self.img_size = img_size
        self.num_ref_chars = num_ref_chars

        self.data_list = [(font_name, char) for font_name in self.font_name_list for char in self.chars_list]

    def __getitem__(self, idx):
        font_name, char = self.data_list[idx]

        cur_ref_char = list(set(self.chars_list).difference(set(char)))
        ref_char_list = np.random.choice(cur_ref_char, size=(self.num_ref_chars,), replace=False)

        font = self.name_font_dict[font_name]
        src = render(self.source, char)
        tgt = render(font, char)
        ref_char = [render(font, char) for char in ref_char_list]

        src_trans = aug_transform(img_size=self.img_size, aug=False)
        tgt_trans = src_trans
        ref_trans = src_trans

        src = src_trans(src)
        tgt = tgt_trans(tgt)
        ref = torch.stack([ref_trans(char) for char in ref_char])

        return {'src': src, 'tgt': tgt, 'ref': ref}

    def __len__(self):
        return len(self.data_list)


def get_train_loader(dataset_path,
                     img_size,
                     num_ref_chars,
                     batch_size,
                     shuffle,
                     num_workers):
    train_dataset = TrainDataset(dataset_path=dataset_path,
                                 img_size=img_size,
                                 num_ref_chars=num_ref_chars)

    return torch.utils.data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)
