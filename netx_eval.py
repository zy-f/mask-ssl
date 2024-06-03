import os
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from imagenet_x import error_ratio, plots, compute_factor_accuracies
from imagenet_x.utils import get_annotation_path, load_annotations, \
                             augment_model_predictions, FACTORS

from dsets.imagenet import ImageNet
from systems import SYSTEMS
from models import ARCHITECTURES
from consts import OUTPUT_DIR, CLASS_SUBSET_100

DL_CONFIG = {
    'batch_size': 128,
    'num_workers': 8,
    'pin_memory': True,
}
PRED_DIR = f'{OUTPUT_DIR}/netx_preds'

### ADAPTED CODE FROM IMAGENET-X
def _load_model_predictions(models_dir, verbose=False):
    filename_label = pd.read_csv(get_annotation_path() / "filename_label.csv")
    _, labels = (
        filename_label.file_name,
        filename_label.set_index("file_name").label,
    )

    models = {}
    top_1_accs = pd.Series(dtype=np.float32)
    model_dirs = os.listdir(models_dir)
    assert len(model_dirs) > 0, "No models found in models_dir"
    for path in tqdm(model_dirs, desc="Loading model predictions", disable=not verbose):
        if not path.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(models_dir, path)).set_index("file_name")
        model = path[:path.find('_')]
        _labels = labels.loc[df.index]
        top_1_accs[model] = (df["predicted_class"] == _labels).mean()
        models[model] = df
    return models, top_1_accs

def _get_factor_accuracies(models_dir, which_factor='top', error_type='class'):
    error_type = {
        "real_class": "is_correct_real",
        "metaclass": "is_metaclass_correct",
        "class": "is_correct",
    }[error_type]
    annotations = load_annotations(which_factor=which_factor, partition='val', filter_prototypes=True)
    model_predictions = _load_model_predictions(models_dir)
    aug_predictions = augment_model_predictions(annotations, model_predictions[0])
    return compute_factor_accuracies(aug_predictions, FACTORS, error_type=error_type)

###

class ImageNetXEvaluator:
    def __init__(self, out_dir=PRED_DIR, save_out=True, dl_config=DL_CONFIG, device='cpu', \
                 subset_classes=None, map_classes=False):
        self.save_out = save_out
        dset = ImageNet('val', subset_classes=subset_classes)
        self.subset_classes = subset_classes
        self.n_cls = len(subset_classes) if subset_classes and map_classes else 1000
        self.dl = DataLoader(dset, shuffle=False, **dl_config)
        self.device = device
        self.out_dir = out_dir
        self.out_files = []
        # self.file_ix = pd.read_csv(get_annotation_path() / "filename_label.csv").values[:,0]
    
    def __del__(self):
        if not self.save_out:
            for f in self.out_files:
                os.remove(f)
    
    def _fix_image_paths(self, fp):
        return fp[:fp.rfind('_')] + fp[fp.rfind('.'):]
    
    # def _reorder(self, df):
    #     return df.set_index('file_name').loc[self.file_ix].reset_index()

    def run_eval(self, ssl_system, arch='resnet50', weight_name='', run_name=None):
        run_name = run_name or f"{ssl_system}_{arch}_{weight_name}"
        model = ARCHITECTURES[arch](num_classes=self.n_cls).to(self.device)
        SYSTEMS[ssl_system].load_weights(model, weight_name=(weight_name or arch))
        preds = {'file_name': [], 'predicted_class': [], 'predicted_probability': []}
        with torch.no_grad():
            for batch in tqdm(self.dl, desc=f"eval {arch} via {ssl_system}"):
                X = batch.x.to(self.device)
                out = model(X)
                out_probs = softmax(out, dim=-1).cpu().numpy()
                pred_cls = out_probs.argmax(axis=-1)
                pred_prob = out_probs[range(len(X)), pred_cls]
                pred_cls = pred_cls.tolist()
                preds['file_name'] += list(map(self._fix_image_paths, batch.path))
                if self.subset_classes:
                    pred_cls = [self.subset_classes[y] for y in pred_cls]
                preds['predicted_class'] += pred_cls
                preds['predicted_probability'] += pred_prob.tolist()
        df = pd.DataFrame(preds)
        # df = self._reorder(df)
        out_path = f"{self.out_dir}/{run_name}.csv"
        df.to_csv(out_path, header=True, index=False)
        self.out_files.append(out_path)

def analyze_all(display=True, savename=None):
    factor_accs = _get_factor_accuracies(PRED_DIR) # df
    error_ratios = error_ratio(factor_accs) # df
    if display:
        print(error_ratios)
        plots.model_comparison(factor_accs.reset_index())
    if savename:
        base = f"{OUTPUT_DIR}/{savename}-"
        error_ratios.to_csv(base+'err_ratios.csv')
        plots.model_comparison(factor_accs.reset_index(), fname=base+"model_comp.png")

def main():
    analyzer = ImageNetXEvaluator(subset_classes=CLASS_SUBSET_100, device='cuda')
    analyzer.run_eval('a2mim')
    analyzer.run_eval('moco_v3')
    # breakpoint()
    analyze_all(savename='baseline_100')

if __name__ == '__main__':
    main()