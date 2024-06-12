import argparse
import os
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from imagenet_x import plots, compute_factor_accuracies
from imagenet_x.utils import get_annotation_path, load_annotations, \
                             augment_model_predictions, FACTORS
from dsets.imagenet import ImageNet

from systems import SYSTEMS
from models import ARCHITECTURES
from consts import OUTPUT_DIR, CLASS_SUBSET_100
FACTOR_SUBSET = ['background', 'color', 'smaller', 'partial_view', 'pattern', 
                 'pose', 'shape', 'subcategory','texture']

DL_CONFIG = {
    'batch_size': 128,
    'num_workers': 8,
    'pin_memory': True,
}
PRED_DIR = f'{OUTPUT_DIR}/netx_preds'

### ADAPTED CODE FROM IMAGENET-X REPO
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
        model = path[:path.find('.')] #path[:path.find('_')]
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
    models, top_1_accs = _load_model_predictions(models_dir)
    print(top_1_accs)
    aug_predictions = augment_model_predictions(annotations, models)
    factor_accs = compute_factor_accuracies(aug_predictions, FACTOR_SUBSET, error_type=error_type)
    factor_accs['val_acc'] = top_1_accs
    print(factor_accs.head(2))
    return factor_accs

def _plot_error_ratios(comparison_df, factors=FACTOR_SUBSET, average_name='average', hue='model', fname=None):
    value = "Error ratio"
    comparison_df = comparison_df[factors + [hue, average_name]].melt(
        id_vars=[hue, average_name], var_name="Factor", value_name="Accuracy"
    )
    comparison_df["Error ratio"] = (1-comparison_df["Accuracy"]) / (1-comparison_df[average_name])
    comparison_df = comparison_df.sort_values(by=value)
    factor_order = (
        comparison_df.groupby("Factor")["Error ratio"].mean().sort_values().index.values
    )
    data_to_display = comparison_df.groupby(["Factor", hue]).mean().reset_index()
    g = plots.plot_bar_plot(data_to_display, x="Factor", y=value, hue=hue, factor_order=factor_order)
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, -1),
        ncol=len(comparison_df[hue].unique())//2,
        title=None,
        frameon=False,
    )
    plt.xticks(rotation=0)
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight", dpi=300, transparent=False)
        plt.close()
### END ADAPTED CODE

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
        print(df.head())
        # df = self._reorder(df)
        out_path = f"{self.out_dir}/{run_name}.csv"
        df.to_csv(out_path, header=True, index=False)
        self.out_files.append(out_path)

def _plot_factor_accs(factor_df, fname='', display=False, eps=0.01,
                     cols=["val_acc", "texture", "subcategory", "partial_view"]):
    factor_df = factor_df.add(eps)
    melted = factor_df[cols].reset_index().melt(id_vars='model', var_name='factor', value_name='accuracy')
    plt.figure(figsize=(20, 3))
    factor_plot = sns.barplot(melted, x='factor', y='accuracy', hue='model')
    plt.xlabel("")
    plt.ylabel('accuracy', fontsize=18)
    plt.ylim(0,1)
    sns.despine()
    sns.move_legend(
        factor_plot,
        "lower left",
        bbox_to_anchor=(0.01, 0.78),
        ncol=len(factor_df)//2,
        title=None,
        frameon=False,
    )
    plt.tight_layout()
    plt.gca().set_axisbelow(True)
    plt.grid(axis="y", which="major", linewidth=1, alpha=0.3, linestyle="--")
    if display:
        plt.show()
    if fname:
        plt.savefig(fname)

def analyze_all(data_dir=PRED_DIR, display=False, savename=None):
    factor_accs = _get_factor_accuracies(data_dir) # df
    error_ratios = (1-factor_accs[FACTOR_SUBSET]).divide((1-factor_accs["average"]), axis=0) # df
    if display:
        print(error_ratios)
        plots.model_comparison(factor_accs.reset_index())
    if savename:
        base = f"{OUTPUT_DIR}/{savename}-"
        print('saving plots and tables')
        factor_accs.to_csv(base+'factor_accs.csv')
        error_ratios.to_csv(base+'err_ratios.csv')
        _plot_factor_accs(factor_accs, fname=base+"factor_acc_comp.png")
        _plot_error_ratios(factor_accs.reset_index(), fname=base+"model_comp.png")

def main(system='ftune', weight='', arch='resnet50', evaluate=False, compare=False, plotname='gen'):
    if evaluate:
        print(f'>>> sys={system}, arch={arch}, weight_name={weight} <<<')
        analyzer = ImageNetXEvaluator(subset_classes=CLASS_SUBSET_100, device='cuda', map_classes=True)
        analyzer.run_eval(system, arch=arch, weight_name=weight,
                            run_name=weight[:-len('-best')])
    if compare:
        analyze_all(savename=plotname, data_dir=f'{OUTPUT_DIR}/netx_preds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate models on imagenet-x')
    parser.add_argument('-c', '--compare', action='store_true')
    parser.add_argument('-n', '--plotname', type=str, help='filename for plot', default='final')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-s', '--system', type=str, help='name of system to use', default='ftune')
    parser.add_argument('-w', '--weight', type=str, help='name of weight to load')
    parser.add_argument('-a', '--arch', type=str, help='name of architecture to load', default='resnet50')
    args = parser.parse_args()
    main(**args.__dict__)
