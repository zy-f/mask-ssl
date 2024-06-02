import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from tqdm import tqdm
import sys
from consts import WEIGHT_DIR
import systems.a2mim_utils as a2u

WEIGHTS = {
    'resnet50': f"{WEIGHT_DIR}/a2mim_r50_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.pth"
}

def load_weights(model, weight_name='resnet50'):
    if weight_name not in WEIGHTS.keys():
        raise ValueError(f'invalid weight name {weight_name}')
    elif weight_name == 'resnet50':
        raw_ckpt = torch.load(WEIGHTS[weight_name])
        sd = {k[k.find('.')+1:] : v for k,v in raw_ckpt['state_dict'].items()}
        model.load_state_dict(sd)

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
    def __init__(self, cfg, model, trn_dset, val_dset=None, device='cpu'):
        backbone = a2u.inject_mask(model, cfg.mask_cfg, layer=cfg.mask_layer)
        self.full_model = a2u.A2MIMModel(backbone).to(device)
        self.trn_dset = a2u.MaskedDataset(trn_dset, mask_gen_cfg=cfg.mask_gen_cfg, mask_layer=cfg.mask_layer)
        self.val_dset = a2u.MaskedDataset(val_dset, mask_gen_cfg=cfg.mask_gen_cfg, mask_layer=cfg.mask_layer)
        self.optimizer = a2u.prepare_optimizer(self.full_model, hparams=cfg.hparams)
        self.loss_func = a2u.A2MIMLoss()
        self.cfg = cfg
        self.device = device

    def train_step(self, batch):
        img = batch.x.to(self.device)
        mask = batch.mask.to(self.device)
        img_rec = self.full_model(img, mask=mask)
        loss = self.loss_func(img, img_rec, mask)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def val_step(self, batch):
        with torch.no_grad():
            img = batch.x.to(self.device)
            mask = batch.mask.to(self.device)
            img_rec = self.full_model(img, mask=mask)
            loss = self.loss_func(img, img_rec, mask)
        return loss.item()

    def save_model(self):
        torch.save(self.full_model.backbone.state_dict(), f'{WEIGHT_DIR}/{self.cfg.exp_name}-best.pth')
    
    def run_training(self):
        trn_dl = DataLoader(self.trn_dset, **self.cfg.dl_kwargs, shuffle=True)
        val_dl = DataLoader(self.val_dset, **self.cfg.dl_kwargs)
        log_out = EasyDict()
        log_out.best_loss = float('inf')
        n_epochs = self.cfg.hparams.n_epochs
        for epoch in range(n_epochs):
            print(f'EPOCH {epoch+1}/{n_epochs}')
            log_out.trn_loss = 0
            log_out.val_loss = 0
            for batch in tqdm(trn_dl, desc='trn', leave=True):
                log_out.trn_loss += self.train_step(batch)
            log_out.trn_loss /= len(self.trn_dset)
            for batch in tqdm(val_dl, desc='val', leave=True):
                log_out.val_loss += self.val_step(batch)
            log_out.val_loss /= len(self.val_dset)

            if log_out.val_loss < log_out.best_loss:
                log_out.best_loss = log_out.val_loss
                self.save_model()
            print(f'trn_loss={log_out.trn_loss}, val_loss={log_out.val_loss}')
