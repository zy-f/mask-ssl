import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import json
from easydict import EasyDict
from consts import CONFIG_DIR, LOG_DIR
import random

### GENERAL
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_config(name, ignore_prefix='_'):
    fp = f"{CONFIG_DIR}/{name}.json"
    with open(fp, 'r') as f:
        config = json.load(f)
    if ignore_prefix:
        config = {k:v for k,v in config.items() if not k.startswith(ignore_prefix)}
    cfg = EasyDict(config)
    return cfg

## METRICS

class LogInput:
    def __init__(self, name, agg_func=lambda x: x):
        self.name = name
        self.agg = agg_func
        self.data = []
    
    def append(self, x):
        self.data.append(x)
    
    def flush(self):
        out = self.agg(self.data) if self.data else None
        self.data = []
        return out

class EpochLogger:
    def __init__(self, inputs):
        self.inputs = {i.name: i for i in inputs}
    
    def add_batch(self, batch_out):
        for k in batch_out.keys():
            self.inputs[k].append(batch_out[k])
    
    def flush(self):
        return {name: self.inputs[name].flush() for name in self.inputs.keys()}


class ScalarMetric:
    def __init__(self, name, input_names, compute_func, report_decimals=4):
        self.name = name
        self.input_names = input_names
        self.compute_func = compute_func
        self.data = []
        self.dec = report_decimals

    def report(self):
        out = f"{self.name}: "
        if len(self.data) < 1:
            out += '-.--'
        else:
            out += f"{round(self.data[-1], self.dec)}"
            if len(self.data) < 2:
                out += " (-.--)"
            else:
                out += f" ({'%+f' % round(self.data[-1] - self.data[-2], self.dec)})"
        return out

    def update(self, *inputs):
        self.data.append(self.compute_func(*inputs))
    
    def last(self):
        return self.data[-1] if len(self.data) > 0 else None

### PLOTTING
class AxisWrapper:
    def __init__(self, ax_obj, ncols):
        self.ax = ax_obj if isinstance(ax_obj, np.ndarray) else np.array([ax_obj])
        self.c = ncols

    def __getitem__(self, i):
        if self.c == 1:
            return self.ax[i]
        return self.ax[i//self.c, i%self.c]

def get_rect_subplots(n, figscale=5):
    ncols = int(np.sqrt(n))
    nrows = int(np.ceil(n/ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*figscale, nrows*figscale))
    fig.set_facecolor('white')
    return fig, AxisWrapper(ax, ncols)

### LOGGING
class LogOutput:
    def __init__(self, inputs, metrics, eval_metric_idx=-1, cmp_func=(lambda new, old: new < old)):
        self.inputs = inputs
        self.metrics = metrics
        self.last_input = None
        self.eval_metric_idx = eval_metric_idx
        self.best_eval = None
        self.cmp_func = cmp_func

    def update(self, inputs):
        self.last_input = inputs
        new_best = False
        for i, m in enumerate(self.metrics):
            m.update(*[inputs[in_name] for in_name in m.input_names])
            if i == self.eval_metric_idx:
                if (self.best_eval is None) or self.cmp_func(m.last(), self.best_eval):
                    self.best_eval = m.last()
                    new_best = True
        return new_best

            
    def pass_log(self): 
        '''
        use for wandb/other logger integration
        passes last computed metrics as {metric_name: value} dict
        '''
        return {m.name: m.last() for m in self.metrics}

    def report(self, joiner='\n'):
        return joiner.join([m.report() for m in self.metrics])

    def plot(self, savepath=None):
        fig, ax = get_rect_subplots(len(self.metrics))
        fig.set_facecolor('lightgrey')
        for i, m in enumerate(self.metrics):
            ax[i].plot(m.data)
            ax[i].set_xlabel('Epoch')
            ax[i].set_ylabel(m.name)
        if savepath:
            plt.savefig(savepath)
    
    def write_out(self, savename):
        df = pd.DataFrame({m.name: m.data for m in self.metrics})
        df.to_csv(f'{LOG_DIR}/{savename}_metrics.csv', header=True, index=True, index_label='epoch')