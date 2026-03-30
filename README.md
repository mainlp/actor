# Active Learning for Hate Speech Classification with Multi-Annotator Models

This repository implements active learning experiments for hate speech classification using BERT-based models. It supports both majority-vote and per-annotator (multi-head) training, with several acquisition strategies for active learning (entropy, BALD, DAL, etc.).

Built on top of [Contextualizing Hate Speech Classifiers with Post-hoc Explanation](https://arxiv.org/abs/2005.02439) (Kennedy et al., ACL 2020).

## Requirements

```bash
conda create -n expl-reg python==3.7.4
conda activate expl-reg
conda install pytorch cudatoolkit -c pytorch
pip install nltk numpy scikit-learn scikit-image matplotlib torchtext
pip install tokenizers==0.0.11 boto3 filelock requests tqdm regex sentencepiece sacremoses
pip install ruamel.yaml mlflow torchsampler
```

## Project Structure

```
├── run_model.py              # Standard BERT fine-tuning (no active learning)
├── run_model_al.py           # BERT fine-tuning with active learning loop
├── acquisitions.py           # Active learning acquisition strategies (entropy, BALD, DAL, etc.)
├── scripts/
│   └── expriment.py          # Main experiment launcher (reads YAML config)
├── bert/                     # BERT modeling, tokenization, optimization
├── hiex/                     # SOC explanation regularization
├── loader/                   # Data loaders (Gab, Stormfront, NYT, annotator-level)
├── utils/
│   ├── config.py             # Default configuration
│   ├── apply_acquisition.py  # Acquisition function dispatch
│   └── utils.py
└── data/                     # Datasets (GHC, identity words)
```

## Running Experiments

All experiments are launched via `scripts/expriment.py`, which reads a YAML config file, overrides parameters from command-line arguments, and calls either `run_model.py` (standard training) or `run_model_al.py` (active learning).

### Basic Usage

```bash
python scripts/expriment.py \
    --mode <baseline|anno> \
    --fold 0 \
    --seed 0 \
    --gpu 0 \
    --experiment_name my_experiment
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `baseline` | `baseline` (majority-vote) or `anno` (per-annotator) |
| `--active_learning` | `False` | Enable active learning loop |
| `--active_strategy` | `None` | Acquisition function: `entropy`, `entropy_norm`, `bald`, `dal`, `random`, `individual_entropy` |
| `--sampling_strategy` | `instance_first` | `instance_first` or `label_first` |
| `--num_heads` | `6` | Number of annotator heads (for multi-head model) |
| `--over_sampling` | `False` | Enable oversampling for class imbalance |
| `--class_weight` | `False` | Use class weights for imbalanced data |
| `--data_path` | `None` | Override the data directory from the YAML config |
| `--seed` | `0` | Random seed |
| `--gpu` | `0` | GPU device ID |
| `--fold` | `0` | Data fold index |
| `--learning_rate` | `1e-5` | Learning rate |
| `--train_batch_size` | `32` | Training batch size |
| `--num_train_epochs` | `10` | Number of training epochs |
| `--init_size` | `60` | Initial labeled pool size (active learning) |
| `--query_sample_size` | `60` | Samples to query per round (active learning) |
| `--rounds` | `100` | Number of active learning rounds |
| `--eval_mode` | `majority` | Evaluation mode: `majority` or `annotation` |

### Example: Active Learning with Annotator Model

```bash
python scripts/expriment.py \
    --mode anno \
    --fold 0 \
    --seed 0 \
    --gpu 0 \
    --active_learning \
    --active_strategy entropy_norm \
    --sampling_strategy label_first \
    --num_heads 6 \
    --over_sampling \
    --class_weight \
    --learning_rate 2e-5 \
    --init_size 60 \
    --query_sample_size 60 \
    --rounds 100 \
    --experiment_name anno_entropy_norm_experiment
```

### Example: Majority-Vote Baseline with Active Learning

```bash
python scripts/expriment.py \
    --mode baseline \
    --fold 0 \
    --seed 0 \
    --gpu 0 \
    --active_learning \
    --active_strategy entropy \
    --over_sampling \
    --learning_rate 2e-5 \
    --experiment_name baseline_entropy_experiment
```

### YAML Config Files

The experiment launcher reads a YAML config file (e.g., `config/anno_act.yaml` or `config/maj_act.yaml`) that specifies default arguments for `run_model_al.py`. Command-line arguments passed to `expriment.py` override the config values. Place your config files under `config/`.

### Tracking with MLflow

Experiments are tracked with [MLflow](https://mlflow.org/). Results are logged automatically. View them with:

```bash
mlflow ui
```

## Data

### Gab Hate Corpus

The full Gab Hate Corpus (GHC) is available at https://osf.io/edua3/. Data files are in `data/majority_gab_dataset_25k/` as train/dev/test.jsonl, where each line is a JSON dict:

```json
{"text_id": 31287737, "Text": "How is that one post not illegal? ...", "im": 0, "cv": 0, "ex": 0, "hd": 0, ...}
```

### HS-Brexit Dataset

Place the HS-Brexit dataset under `data/data_post-competition/HS-Brexit_dataset/` with subdirectories for annotation-level and majority-vote splits.

## Reference

If you find this code helpful, please cite:

```bibtex
@inproceedings{kennedy2020contextualizing,
   author = {Kennedy*, Brendan and Jin*, Xisen and Mostafazadeh Davani, Aida and Dehghani, Morteza and Ren, Xiang},
   title = {Contextualizing {H}ate {S}peech {C}lassifiers with {P}ost-hoc {E}xplanation},
   year = {2020},
   booktitle = {Proceedings of the 58th {A}nnual {M}eeting of the {A}ssociation for {C}omputational {L}inguistics}
}
```
