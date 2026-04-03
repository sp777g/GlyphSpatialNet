# GlyphSpatialNet
This is the official repository for the CVPR 2026 paper "Rethinking Glyph Spatial Information in Font Generation".

## Dataset
Please refer to this [repository](https://github.com/sp777g/GlyphSpatialNet_Dataset) to build the dataset.

## Conda Environment
```
conda create -n FFG python=3.12 -y && conda activate FFG

pip --default-timeout=99999 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm tensorboard lpips torchmetrics einops
```

## Training & Testing

```
sh script_main.sh
```

## Evaluation

```
sh script_eval.sh
sh script_eval_norm.sh
```

## Acknowledgments
Our code references the following open-source projects:
 - [RDDM](https://github.com/nachifur/RDDM)
 - [MSD-Font](https://github.com/fubinfb/MSD-Font)
 - [FFG](https://github.com/clovaai/fewshot-font-generation)

We thank the authors for their excellent work.
