import os
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from imagenet_x import get_factor_accuracies, error_ratio, plots
from imagenet_x.utils import get_annotation_path

from dsets.imagenet import ImageNet
from systems import SYSTEMS
from models import ARCHITECTURES
from consts import OUTPUT_DIR

DL_CONFIG = {
    'batch_size': 128,
    'num_workers': 8,
    'pin_memory': True,
}
PRED_DIR = f'{OUTPUT_DIR}/netx_preds'

class ImageNetXEvaluator:
    def __init__(self, out_dir=PRED_DIR, save_out=True, dl_config=DL_CONFIG, device='cpu'):
        self.save_out = save_out
        dset = ImageNet('val')
        self.dl = DataLoader(dset, shuffle=False, **dl_config)
        self.device = device
        self.out_dir = out_dir
        self.out_files = []
        self.file_ix = pd.read_csv(get_annotation_path() / "filename_label.csv").values[:,0]
    
    def __del__(self):
        if not self.save_out:
            for f in self.out_files:
                os.remove(f)
    
    def _fix_image_paths(self, fp):
        return fp[:fp.rfind('_')] + fp[fp.rfind('.'):]
    
    def _reorder(self, df):
        return df.set_index('file_name').loc[self.file_ix].reset_index()

    def run_eval(self, ssl_system, arch='resnet50', weight_name='', run_name=None):
        run_name = run_name or f"{ssl_system}_{arch}_{weight_name}"
        model = ARCHITECTURES[arch]().to(self.device)
        SYSTEMS[ssl_system].load_weights(model, weight_name=(weight_name or arch))
        preds = {'file_name': [], 'predicted_class': [], 'predicted_probability':[]}
        with torch.no_grad():
            for batch in tqdm(self.dl, desc=f"eval {arch} via {ssl_system}"):
                X = batch.x.to(self.device)
                out = model(X)
                out_probs = softmax(out, dim=-1).cpu().numpy()
                pred_cls = out_probs.argmax(axis=-1)
                pred_prob = out_probs[range(len(X)), pred_cls]
                preds['file_name'] += list(map(self._fix_image_paths, batch.path))
                preds['predicted_class'] += pred_cls.tolist()
                preds['predicted_probability'] += pred_prob.tolist()
        df = pd.DataFrame(preds)
        df = self._reorder(df)
        out_path = f"{self.out_dir}/{run_name}.csv"
        df.to_csv(out_path, header=True, index=False)
        self.out_files.append(out_path)

def analyze_all(display=True, savename=None):
    factor_accs = get_factor_accuracies(PRED_DIR) # df
    error_ratios = error_ratio(factor_accs) # df
    if display:
        print(error_ratios)
        plots.model_comparison(factor_accs.reset_index())
    if savename:
        base = f"{OUTPUT_DIR}/{savename}-"
        error_ratios.to_csv(base+'err_ratios.csv')
        plots.model_comparison(factor_accs.reset_index(), fname=base+"model_comp.png")

def main():
    # analyzer = ImageNetXEvaluator(device='cuda')
    # analyzer.run_eval('a2mim')
    # analyzer.run_eval('moco_v3')
    # breakpoint()
    analyze_all(savename='milestone')

if __name__ == '__main__':
    main()