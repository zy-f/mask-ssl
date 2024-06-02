import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from easydict import EasyDict

from consts import WEIGHT_DIR
from utils import ScalarMetric, LogOutput, LogInput, EpochLogger

class GenericFinetuneSystem:
    def __init__(self, cfg, model, trn_dset, val_dset, tst_dset, device='cpu'):
        self.model = model
        train_params = []
        if cfg.get('tune_layers'):
            for name, param in model.named_parameters():
                main_layer = name[:name.find('.')]
                if main_layer in cfg.tune_layers:
                    print(f'adding {name} to trainable params')
                    param.requires_grad = True
                    train_params.append(param)
                else:
                    param.requires_grad = False
        else:
            train_params = model.parameters()
                
        model.to(device)
        self.trn_dset = trn_dset
        self.val_dset = val_dset
        self.tst_dset = tst_dset
        self.device = device
        self.cfg = cfg
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(train_params, 
                                           lr=cfg.hparams.lr,
                                           weight_decay=cfg.hparams.wd)
        ## logging
        dummy = lambda x: x
        get_acc = lambda preds, ys: np.mean(np.argmax(preds, axis=1) == ys)
        inputs = [
            LogInput('trn_loss', agg_func=np.mean),
            LogInput('trn_pred', agg_func=np.concatenate),
            LogInput('trn_y', agg_func=np.concatenate),
            LogInput('val_loss', agg_func=np.mean),
            LogInput('val_pred', agg_func=np.concatenate),
            LogInput('val_y', agg_func=np.concatenate),
        ]
        self.epoch_logger = EpochLogger(inputs)
        metrics = [
            ScalarMetric('trn loss', input_names=['trn_loss'], compute_func=dummy),
            ScalarMetric('trn acc', input_names=['trn_pred', 'trn_y'], compute_func=get_acc),
            ScalarMetric('val loss', input_names=['val_loss'], compute_func=dummy),
            ScalarMetric('val acc', input_names=['val_pred', 'val_y'], compute_func=get_acc),
        ]
        self.log_outputs = LogOutput(inputs, metrics, eval_metric_idx=2)
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        y_pred = self.model(x)
        loss = self.loss_func(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return EasyDict(
            trn_loss = loss.item(),
            trn_pred = y_pred.detach().cpu().numpy(),
            trn_y = batch.y.detach().numpy()
        )
    
    def val_step(self, batch):
        with torch.no_grad():
            x = batch.x.to(self.device)
            y = batch.y.to(self.device)
            y_pred = self.model(x)
            loss = self.loss_func(y_pred, y)
        return EasyDict(
            val_loss = loss.item(),
            val_pred = y_pred.detach().cpu().numpy(),
            val_y = batch.y.detach().numpy()
        )

    def test_step(self, batch):
        out = self.val_step(batch)
        return EasyDict(tst_loss = out.val_loss, tst_pred = out.val_pred, tst_y = out.val_y)

    def save_model(self):
        torch.save(self.model.state_dict(), f'{WEIGHT_DIR}/{self.cfg.exp_name}-best.pth')
    
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
            if new_best:
                print('new best achieved, saving')
                self.save_model()
            print(self.log_outputs.report())
        self.log_outputs.write_out(savename=self.cfg.exp_name)
