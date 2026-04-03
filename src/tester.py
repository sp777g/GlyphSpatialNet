import os
import json
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from pathlib import Path

try:
    from datasets.utils import aug_transform, tensor_to_pil, get_ttf_paths, open_font, render
    from src.model import Sampler, DownSampleEncoder
except ImportError:
    from FFG.datasets.utils import aug_transform, tensor_to_pil, get_ttf_paths, open_font, render
    from FFG.src.model import Sampler, DownSampleEncoder


def get_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class Tester(object):
    def __init__(
            self,
            model,
            style_enc,
            up_dec,
            image_size,
            down_size,
            out_dir,
            path_gen_chars,
            path_ref_chars,
            path_source,
            path_ttf,
            path_ckpt,
            sampling_time_steps,
            sample_batch_size,
            self_condition=False,
            time_steps=1000,
    ):
        self.device = get_device()
        self.image_size = image_size

        self.sampler = Sampler(
            device=self.device,
            time_steps=time_steps,
            sampling_time_steps=sampling_time_steps,
            self_condition=self_condition,
        )
        self.sample_batch_size = sample_batch_size

        self.model = model.to(self.device)
        self.style_enc = style_enc.to(self.device)
        self.up_dec = up_dec.to(self.device)

        self.down_enc = DownSampleEncoder(img_size=down_size).to(self.device)

        ckpt = torch.load(path_ckpt, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model'])
        self.style_enc.load_state_dict(ckpt['style_enc'])
        self.up_dec.load_state_dict(ckpt['up_dec'])

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.style_enc = torch.nn.DataParallel(self.style_enc)
            self.up_dec = torch.nn.DataParallel(self.up_dec)

        self.model.eval()
        self.style_enc.eval()
        self.up_dec.eval()

        ##################################################################

        self.out_dir = out_dir

        self.source = open_font(path_source)
        self.target_path_list = sorted(get_ttf_paths(path_ttf))
        self.target_list = [(Path(path).stem, open_font(path)) for path in self.target_path_list]

        self.ref_char_list = json.load(open(path_ref_chars, encoding='utf-8'))
        self.chars_list = json.load(open(path_gen_chars, encoding='utf-8'))

    @torch.no_grad()
    def test(self):
        for font_name, font in tqdm(self.target_list):
            gen_dir = os.path.join(self.out_dir, 'genimgs', font_name)
            gt_dir = os.path.join(self.out_dir, 'gtimgs', font_name)

            if not os.path.exists(gen_dir):
                os.makedirs(gen_dir, exist_ok=True)

            if not os.path.exists(gt_dir):
                os.makedirs(gt_dir, exist_ok=True)

            for idx, char in enumerate(self.chars_list):
                gt_img = render(font, char)
                gt_img.save(os.path.join(gt_dir, f'{idx:4d}.png'))

            trans = aug_transform(img_size=self.image_size, aug=False)

            ref_char = [render(font, char) for char in self.ref_char_list]
            ref = torch.stack([trans(char) for char in ref_char]).unsqueeze(0)
            ref_h = ref.to(self.device)
            style_emb = self.style_enc(ref_h.flatten(0, 1))
            b, c, h, w = style_emb.shape
            style_emb = style_emb.view(b // 8, 8, c, h, w)  # ref_chars_for_test_8.json
            style_emb = torch.mean(style_emb, dim=1)

            style_emb = style_emb.repeat(self.sample_batch_size, 1, 1, 1)

            batch_src = list()
            for idx, char in enumerate(self.chars_list):
                src = render(self.source, char)
                src = trans(src)
                batch_src.append(src)

            batch_src = torch.stack(batch_src)
            batches_src = torch.split(batch_src, self.sample_batch_size)

            idx = 0
            for src in batches_src:
                src_h = src.to(self.device)

                src = self.down_enc(src_h)
                src_theta = src

                pred = self.sampler.DDIM_sample(self.model, src_theta, style_emb)

                pred_h = self.up_dec(pred, style_emb)

                for img in pred_h:
                    img = tensor_to_pil(img.unsqueeze(0), padding=0, n_row=1)
                    img.save(os.path.join(gen_dir, f'{idx:4d}.png'))
                    idx -= -1


def test(
        model,
        style_enc,
        up_dec,
        image_size,
        down_size,
        out_dir,
        path_gen_chars,
        path_ref_chars,
        path_source,
        path_ttf,
        path_ckpt,
):
    sampling_time_steps = 5
    sample_batch_size = 200

    tester = Tester(
        model,
        style_enc,
        up_dec,
        image_size,
        down_size,
        out_dir,
        path_gen_chars,
        path_ref_chars,
        path_source,
        path_ttf,
        path_ckpt,
        sampling_time_steps=sampling_time_steps,
        sample_batch_size=sample_batch_size,
    )

    tester.test()
    torch.cuda.empty_cache()
