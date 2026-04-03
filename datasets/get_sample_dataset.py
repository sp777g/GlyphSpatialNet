import json
import os
import numpy as np
import torch

from torch.utils.data import Dataset

try:
    from datasets.utils import aug_transform, get_ttf_paths, open_font, render
except ImportError:
    from FFG.datasets.utils import aug_transform, get_ttf_paths, open_font, render


class SampleDataset(Dataset):
    def __init__(self, dataset_path, img_size):
        self.source_path = os.path.join(dataset_path, 'SourceHanSerifSC-Regular.otf')
        self.source = open_font(self.source_path)

        self.target_path_list = get_ttf_paths(os.path.join(dataset_path, 'unseen_fonts_for_test'))
        self.target_list = [open_font(path) for path in self.target_path_list]

        self.chars_list = json.load(open(os.path.join(dataset_path, 'unseen_chars_for_test.json'), encoding='utf-8'))
        self.ref_char_list = json.load(open(os.path.join(dataset_path, 'ref_chars_for_test_8.json'), encoding='utf-8'))

        self.img_size = img_size

    def __getitem__(self, _):
        font = np.random.choice(self.target_list)
        char = np.random.choice(self.chars_list)

        src = render(self.source, char)
        tgt = render(font, char)
        ref_char = [render(font, char) for char in self.ref_char_list]

        trans = aug_transform(img_size=self.img_size, aug=False)

        src = trans(src)
        tgt = trans(tgt)
        ref = torch.stack([trans(char) for char in ref_char])

        return {'src': src, 'tgt': tgt, 'ref': ref}

    def __len__(self):
        return 1024


def get_sample_loader(dataset_path,
                      img_size,
                      batch_size,
                      shuffle,
                      num_workers):
    train_dataset = SampleDataset(dataset_path=dataset_path,
                                  img_size=img_size)

    return torch.utils.data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)
