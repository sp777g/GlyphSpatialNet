"""
conda create -n FFG python=3.12 -y && conda activate FFG

pip --default-timeout=99999 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm tensorboard lpips torchmetrics einops
"""

import torch
import os

from src import set_seed
from src import ResidualDiffusion, train_stage_1
from src import StyleEncoder, UpSampleDecoder, train_stage_2
from src import test


def get_folder():
    if os.path.isdir('../data'):
        return '../data'
    elif os.path.isdir('/home/sp/data'):
        return '/home/sp/data'
    else:
        exit('Dataset Not Found!')


def merge_model_to_single_ckpt(
        ckpt_path_model,
        ckpt_path_style_enc,
        ckpt_path_up_dec,
        save_path
):
    ckpt_model = torch.load(ckpt_path_model, weights_only=True)
    ckpt_style_enc = torch.load(ckpt_path_style_enc, weights_only=True)
    ckpt_up_dec = torch.load(ckpt_path_up_dec, weights_only=True)

    ckpt = {
        'model': ckpt_model,
        'style_enc': ckpt_style_enc,
        'up_dec': ckpt_up_dec
    }

    torch.save(ckpt, save_path)


def main(
        debug_flag=False,
        train_flag=True,
        test_flag=True,
        gpu_id=(0,)
):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_id))
    set_seed(42)
    dataset_folder = get_folder()

    image_size = 128
    down_size = 64
    num_ref = 3

    rd_params = {
        'dim': 48,
        'dim_multi': (1, 2, 4, 8),
        'img_channels': 1,
        'resnet_block_groups': 8
    }
    sia_params = {
        'dim': 8,
        'dim_multi': (1, 2, 4, 8),
        'img_channels': 1,
        'resnet_block_groups': 2,
    }
    model = ResidualDiffusion(rd_params=rd_params, sia_params=sia_params)

    style_enc_params = {
        'dim': 32,
        'dim_multi': (1, 2, 4, 8),
        'img_channels': 1,
        'resnet_block_groups': 4,
    }
    style_enc = StyleEncoder(**style_enc_params)

    up_dec_params = {
        'dim': 32,
        'dim_multi': (1, 2, 4, 8),
        'img_channels': 1,
        'resnet_block_groups': 4,
        'img_size': image_size
    }
    up_dec = UpSampleDecoder(**up_dec_params)

    if train_flag:
        stage_1_train_steps = 800000
        stage_1_train_batch_size = 64

        train_stage_1(
            model=model,
            style_enc=style_enc,
            dataset_folder=dataset_folder,
            image_size=image_size,
            down_size=down_size,
            num_ref_chars=num_ref,
            train_steps=stage_1_train_steps,
            train_batch_size=stage_1_train_batch_size,
            debug=debug_flag
        )

        stage_2_train_steps = 50000
        stage_2_train_batch_size = 64

        train_stage_2(
            up_dec=up_dec,
            style_enc=style_enc,
            ckpt_path_style_enc='./results/train_stage_1/style_enc.pth',
            dataset_folder=dataset_folder,
            image_size=image_size,
            down_size=down_size,
            num_ref_chars=num_ref,
            train_steps=stage_2_train_steps,
            train_batch_size=stage_2_train_batch_size,
            debug=debug_flag
        )

        merge_model_to_single_ckpt(
            ckpt_path_model='./results/train_stage_1/model.pth',
            ckpt_path_style_enc='./results/train_stage_1/style_enc.pth',
            ckpt_path_up_dec='./results/train_stage_2/up_dec.pth',
            save_path='./results/ckpt.pth'
        )
    else:
        print('Skip the training process')

    if test_flag:
        test(
            model=model,
            style_enc=style_enc,
            up_dec=up_dec,
            image_size=image_size,
            down_size=down_size,
            out_dir='./results/test_UFSC',
            path_gen_chars='../data/dataset_test_UFSC/seen_chars_for_test.json',
            path_ref_chars='../data/dataset_test_UFSC/ref_chars_for_test_8.json',
            path_source='../data/dataset_test_UFSC/SourceHanSerifSC-Regular.otf',
            path_ttf='../data/dataset_test_UFSC/unseen_fonts_for_test',
            path_ckpt='results/ckpt.pth'
        )
        test(
            model=model,
            style_enc=style_enc,
            up_dec=up_dec,
            image_size=image_size,
            down_size=down_size,
            out_dir='./results/test_UFUC',
            path_gen_chars='../data/dataset_test_UFUC/unseen_chars_for_test.json',
            path_ref_chars='../data/dataset_test_UFUC/ref_chars_for_test_8.json',
            path_source='../data/dataset_test_UFUC/SourceHanSerifSC-Regular.otf',
            path_ttf='../data/dataset_test_UFUC/unseen_fonts_for_test',
            path_ckpt='results/ckpt.pth'
        )
    else:
        print('Skip the testing process')


if __name__ == "__main__":
    main()
