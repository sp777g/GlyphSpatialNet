import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

try:
    from datasets import get_train_loader, get_sample_loader, tensor_to_pil
    from src.model import get_coefficient, Sampler, DownSampleEncoder
except ImportError:
    from FFG.datasets import get_train_loader, get_sample_loader, tensor_to_pil
    from FFG.src.model import get_coefficient, Sampler, DownSampleEncoder


def cycle(dl):
    while True:
        for data in dl:
            yield data


# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# trainer class
class Trainer(object):
    def __init__(
            self,
            model,
            style_enc,
            dataset_folder,
            project_folder,
            image_size,
            down_size,
            num_ref_chars,
            lr,
            train_steps,
            train_batch_size,
            save_every,
            sampling_time_steps,
            sample_batch_size,
            self_condition=False,
            time_steps=1000,
    ):
        super().__init__()
        assert torch.cuda.is_available()
        self.device = torch.device('cuda:0')

        self.model = model.to(self.device)
        self.style_enc = style_enc.to(self.device)

        self.down_enc = DownSampleEncoder(img_size=down_size).to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.style_enc = torch.nn.DataParallel(self.style_enc)
            self.down_enc = torch.nn.DataParallel(self.down_enc)

        self.image_size = image_size
        self.down_size = down_size
        self.num_ref_chars = num_ref_chars

        self.train_loader = cycle(get_train_loader(
            dataset_path=os.path.join(dataset_folder, 'dataset_train'),
            img_size=self.image_size,
            num_ref_chars=num_ref_chars,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0
        ))

        self.sample_loader = cycle(get_sample_loader(
            dataset_path=os.path.join(dataset_folder, 'dataset_test_UFUC'),
            img_size=self.image_size,
            batch_size=sample_batch_size,
            shuffle=True,
            num_workers=0
        ))

        self.time_steps = time_steps
        self.sampling_time_steps = sampling_time_steps
        assert self.sampling_time_steps <= self.time_steps

        self.self_condition = self_condition

        coefficient = get_coefficient(time_steps=self.time_steps)
        self.alphas_cumsum = coefficient['alphas_cumsum'].to(self.device)
        self.betas_cumsum = coefficient['betas_cumsum'].to(self.device)

        self.sampler = Sampler(
            device=self.device,
            time_steps=self.time_steps,
            sampling_time_steps=self.sampling_time_steps,
            self_condition=self.self_condition,
        )

        self.train_steps = train_steps
        self.save_every = save_every
        self.project_folder = project_folder

        # optimizer
        self.opt = AdamW([
            {'params': self.model.parameters()},
            {'params': self.style_enc.parameters()},
        ], lr=lr, weight_decay=1e-3)

        # scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.opt,
            num_warmup_steps=min(10000, self.train_steps // 10),
            num_training_steps=self.train_steps
        )

        # step counter state
        self.step = 0

        if os.path.isdir(os.path.join(self.project_folder, 'ckpt')):
            ckpt_list = [f for f in os.listdir(os.path.join(self.project_folder, 'ckpt')) if f.endswith('.pth')]
            if len(ckpt_list) > 0:
                ckpt_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                ckpt_path = os.path.join(self.project_folder, 'ckpt', ckpt_list[-1])
                self.load(ckpt_path)

        if not os.path.exists(self.project_folder):
            os.makedirs(self.project_folder, exist_ok=True)

        self.writer = None

    def q_sample(self, x_start, x_res, t, noise):
        return (
                x_start + extract(self.alphas_cumsum, t, x_start.shape) * x_res +
                extract(self.betas_cumsum, t, x_start.shape) * noise
        )

    def p_losses(self, data):
        src = data['src'].to(self.device)
        tgt = data['tgt'].to(self.device)
        ref = data['ref'].to(self.device)

        src = self.down_enc(src)
        tgt = self.down_enc(tgt)

        style_cond = self.style_enc(ref.flatten(0, 1))
        b, c, h, w = style_cond.shape
        style_cond = style_cond.view(b // self.num_ref_chars, self.num_ref_chars, c, h, w)
        style_cond = torch.mean(style_cond, dim=1)

        src_theta = src

        b, *_ = src_theta.shape
        t = torch.randint(0, self.time_steps, (b,), device=self.device).long()
        noise = torch.randn_like(src_theta)

        x_res = src_theta - tgt

        # noise sample
        x_t = self.q_sample(tgt, x_res, t, noise)

        model_out = self.model(x_t, t, src_theta, style_cond)

        loss = dict()
        loss['noise'] = F.mse_loss(model_out['noise'], noise)
        loss['x_0'] = F.mse_loss(model_out['x_0'], tgt)

        return loss

    def train(self):
        if self.step >= self.train_steps:
            print('Stage 1 training completed')
            return

        self.writer = SummaryWriter(os.path.join(self.project_folder, 'logs'))

        self.model.train()
        self.style_enc.train()

        with tqdm(initial=self.step, total=self.train_steps, ncols=150) as pbar:
            while self.step < self.train_steps:
                data = next(self.train_loader)

                self.opt.zero_grad()

                loss = self.p_losses(data)
                total_loss = loss['noise'] + loss['x_0']
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.style_enc.parameters(), 1.0)

                self.opt.step()
                self.scheduler.step()

                self.step += 1

                if self.step != 0 and self.step % (max(1, self.save_every // 10)) == 0:
                    torch.cuda.empty_cache()
                    self.sample()
                    torch.cuda.empty_cache()

                if self.step != 0 and self.step % self.save_every == 0:
                    self.save()

                total_loss = total_loss.item()
                loss_noise = loss['noise'].item()
                loss_x_0 = loss['x_0'].item()
                cur_lr = self.scheduler.get_last_lr()[0]

                des = '|'.join([
                    f'total={total_loss:.4f}',
                    f'noise={loss_noise:.4f}',
                    f'x_0={loss_x_0:.4f}',
                    f'lr={cur_lr:.4e}'
                ])

                pbar.set_description(des)

                self.writer.add_scalar('Train/total_loss', total_loss, self.step)
                self.writer.add_scalar('Train/loss_noise', loss_noise, self.step)
                self.writer.add_scalar('Train/loss_x_0', loss_x_0, self.step)
                self.writer.add_scalar('Train/lr', cur_lr, self.step)

                pbar.update(1)

        if torch.cuda.device_count() > 1:
            model_state_dict = self.model.module.state_dict()
            style_enc_state_dict = self.style_enc.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            style_enc_state_dict = self.style_enc.state_dict()

        torch.save(model_state_dict, os.path.join(self.project_folder, 'model.pth'))
        torch.save(style_enc_state_dict, os.path.join(self.project_folder, 'style_enc.pth'))

        print('Stage 1 training completed')

    @torch.no_grad()
    def sample(self):
        self.model.eval()
        self.style_enc.eval()

        data = next(self.sample_loader)
        src = data['src'].to(self.device)
        ref = data['ref'].to(self.device)
        tgt = data['tgt'].to(self.device)

        src = self.down_enc(src)
        tgt = self.down_enc(tgt)

        style_cond = self.style_enc(ref.flatten(0, 1))
        b, c, h, w = style_cond.shape
        style_cond = style_cond.view(b // 8, 8, c, h, w)
        style_cond = torch.mean(style_cond, dim=1)

        src_theta = src

        pred = self.sampler.DDIM_sample(self.model, src_theta, style_cond)

        src_h = F.interpolate(src, size=[self.image_size, self.image_size],
                              mode='bilinear', align_corners=True, antialias=True)
        pred_h = F.interpolate(pred, size=[self.image_size, self.image_size],
                               mode='bilinear', align_corners=True, antialias=True)
        tgt_h = F.interpolate(tgt, size=[self.image_size, self.image_size],
                              mode='bilinear', align_corners=True, antialias=True)

        all_images_list = list()
        for src, ref, pred, tgt in zip(src_h, ref, pred_h, tgt_h):
            fake = pred * 0.5 + 0.5
            real = tgt * 0.5 + 0.5
            res_p = torch.nn.functional.relu(real - fake)
            res_m = torch.nn.functional.relu(fake - real)
            res_p = torch.cat([torch.zeros_like(res_p), res_p, res_p], dim=0)
            res_m = torch.cat([res_m, torch.zeros_like(res_m), res_m], dim=0)
            res = (1. - (res_p + res_m)) * 2 - 1

            row = torch.cat([
                src.unsqueeze(0).repeat(1, 3, 1, 1),
                ref.repeat(1, 3, 1, 1),
                pred.unsqueeze(0).repeat(1, 3, 1, 1),
                tgt.unsqueeze(0).repeat(1, 3, 1, 1),
                res.unsqueeze(0)
            ], dim=0)
            all_images_list.append(row)

        all_images = torch.cat(all_images_list, dim=0)
        file_name = f'sample-{self.step}.png'
        all_images = tensor_to_pil(all_images, padding=0, n_row=3 + 8 + 1, img_type='RGB')

        save_path = os.path.join(self.project_folder, 'samples')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        all_images.save(os.path.join(save_path, file_name))

        self.model.train()
        self.style_enc.train()

    def save(self):
        if torch.cuda.device_count() > 1:
            model_state_dict = self.model.module.state_dict()
            style_enc_state_dict = self.style_enc.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            style_enc_state_dict = self.style_enc.state_dict()

        data = {
            'model': model_state_dict,
            'style_enc': style_enc_state_dict,
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.step
        }

        if not os.path.exists(os.path.join(self.project_folder, 'ckpt')):
            os.makedirs(os.path.join(self.project_folder, 'ckpt'), exist_ok=True)

        torch.save(data, os.path.join(self.project_folder, 'ckpt', f'ckpt_{self.step}.pth'))

        print(f'\nSuccessfully saved model-{self.step}\n')

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(ckpt['model'])
        self.style_enc.load_state_dict(ckpt['style_enc'])
        self.opt.load_state_dict(ckpt['opt'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.step = ckpt['step']

        print(f'\nModel-{self.step} loaded\n')


def train_stage_1(
        model,
        style_enc,
        dataset_folder,
        image_size,
        down_size,
        num_ref_chars,
        train_steps,
        train_batch_size,
        debug=False
):
    project_folder = './results/train_stage_1'

    lr = 5e-5

    train_steps = train_steps if not debug else 20
    train_batch_size = train_batch_size
    save_every = 10000 if not debug else 10

    sampling_time_steps = 5
    sample_batch_size = 25

    trainer = Trainer(
        model=model,
        style_enc=style_enc,
        dataset_folder=dataset_folder,
        project_folder=project_folder,
        image_size=image_size,
        down_size=down_size,
        num_ref_chars=num_ref_chars,
        lr=lr,
        train_steps=train_steps,
        train_batch_size=train_batch_size,
        save_every=save_every,
        sampling_time_steps=sampling_time_steps,
        sample_batch_size=sample_batch_size
    )

    # train
    trainer.train()
    torch.cuda.empty_cache()
