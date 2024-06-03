import torch
import numpy as np
from torch.utils.data import DataLoader
from easydict import EasyDict
from tqdm import tqdm
import sys
from consts import WEIGHT_DIR
from utils import ScalarMetric, LogOutput, LogInput, EpochLogger
import systems.a2mim_utils as a2u

WEIGHTS = {
    'resnet50': f"{WEIGHT_DIR}/a2mim_r50_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.pth"
}

def load_weights(model, weight_name='resnet50', delete_keys=[]):
    if weight_name not in WEIGHTS.keys():
        raise ValueError(f'invalid weight name {weight_name}')
    elif weight_name == 'resnet50':
        raw_ckpt = torch.load(WEIGHTS[weight_name])
        sd = {k[k.find('.')+1:] : v for k,v in raw_ckpt['state_dict'].items()}
        for dk in delete_keys:
            del sd[dk]
        print(model.load_state_dict(sd, strict=False))

class A2MIMSystem:
    """
    - cfg: easydict
        - exp_name: str
        - mask_cfg: nested dict
            - mask_mode: str
            - mask_init: float
            - replace: bool
            - detach: bool
        - mask_gen_cfg: nested dict
        - mask_layer: int (1 to 4)
        - hparams: nested dict
            - n_epochs: int
            - lr: float
            - wd: float
        - dl_kwargs: nested dict of args for dataloader
        - opt_kwargs: nested dict of optimizer args (learning rate, etc.)
    - model: unmodified resnet model
    - trn_dset, val_dset are pytorch datasts
    - Optimizer: class of optimizer to use (e.g. torch.optim.AdamW)
    """
    def __init__(self, cfg, model, trn_dset, val_dset, device='cpu'):
        backbone = a2u.inject_mask(model, cfg.mask_cfg, layer=cfg.mask_layer)
        self.full_model = a2u.A2MIMModel(backbone).to(device)
        self.trn_dset = a2u.MaskedDataset(trn_dset, mask_gen_cfg=cfg.mask_gen_cfg, mask_layer=cfg.mask_layer)
        self.val_dset = a2u.MaskedDataset(val_dset, mask_gen_cfg=cfg.mask_gen_cfg, mask_layer=cfg.mask_layer)
        self.loss_func = a2u.A2MIMLoss()
        self.optimizer = a2u.prepare_optimizer(self.full_model, hparams=cfg.hparams)
        self.cfg = cfg
        self.device = device
        # logging
        inputs = [
            LogInput('trn_loss', agg_func=np.mean),
            LogInput('val_loss', agg_func=np.mean),
        ]
        self.epoch_logger = EpochLogger(inputs)
        dummy = lambda x: x
        metrics = [
            ScalarMetric('trn loss', input_names=['trn_loss'], compute_func=dummy),
            ScalarMetric('val loss', input_names=['val_loss'], compute_func=dummy)
        ]
        self.log_outputs = LogOutput(inputs, metrics, eval_metric_idx=1)

    def train_step(self, batch):
        self.optimizer.zero_grad()
        img_raw = batch.x.to(self.device)
        img_mask = batch.x_mask.to(self.device)
        mask = batch.mask.to(self.device)
        img_rec = self.full_model(img_mask, mask=mask)
        loss = self.loss_func(img_raw, img_rec, mask)
        loss.backward()
        self.optimizer.step()
        return EasyDict(trn_loss=loss.item())
    
    def val_step(self, batch):
        with torch.no_grad():
            img_raw = batch.x.to(self.device)
            img_mask = batch.x_mask.to(self.device)
            mask = batch.mask.to(self.device)
            img_rec = self.full_model(img_mask, mask=mask)
            loss = self.loss_func(img_raw, img_rec, mask)
        return EasyDict(val_loss=loss.item())

    def save_model(self):
        torch.save(self.full_model.backbone.state_dict(), f'{WEIGHT_DIR}/{self.cfg.exp_name}-best.pth')
    
    def run_training(self):
        trn_dl = DataLoader(self.trn_dset, **self.cfg.dl_kwargs, shuffle=True)
        val_dl = DataLoader(self.val_dset, **self.cfg.dl_kwargs)
        self.epoch_logger.flush()
        n_epochs = self.cfg.hparams.n_epochs
        for epoch in range(n_epochs):
            print(f'EPOCH {epoch+1}/{n_epochs}')
            for batch in tqdm(trn_dl, desc='trn', leave=True):
                batch_out = self.train_step(batch)
                self.epoch_logger.add_batch(batch_out)
            for batch in tqdm(val_dl, desc='val', leave=True):
                batch_out = self.val_step(batch)
                self.epoch_logger.add_batch(batch_out)
            epoch_data = self.epoch_logger.flush()
            new_best = self.log_outputs.update(epoch_data)
            print(self.log_outputs.report())
            if new_best:
                print('new best achieved, saving')
                self.save_model()
            self.log_outputs.write_out(savename=self.cfg.exp_name)