# Video Action Recognition

This repository contains a cleaned and improved video action recognition training
pipeline. It keeps the original LRCN-style model, fixes split and sequence
handling bugs in the base code, and adds Weights & Biases logging for
train/validation/test evidence.

## Setup

Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

## Dataset Layout

The training code expects extracted frame folders:

```text
UCF50_frames/
    BaseballPitch/
        v_BaseballPitch_g01_c01/
            frame_0000.jpg
            frame_0001.jpg
            ...
```

If you only have raw videos, extract uniformly sampled frames first:

```bash
python preprocess.py --input_dir UCF50 --output_dir UCF50_frames --frames_per_video 16
```

## Run Training

```bash
python run.py \
    --frame_dir UCF50_frames \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --cnn_backbone resnet34 \
    --pretrained true \
    --bidirectional true \
    --attention_pooling true \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode train \
    --n_epochs 30 \
    --output_dir outputs
```

To log required evidence to Weights & Biases:

```bash
wandb login
python run.py --frame_dir UCF50_frames --n_classes 50 --batch_size 4 --mode train --use_wandb true
```

Training writes:

- `models/best_model_wts.pt`
- `splits.npy`
- `outputs/training_history.json`

## Run Evaluation

```bash
python run.py \
    --ckpt models/best_model_wts.pt \
    --model_type lrcn \
    --cnn_backbone resnet34 \
    --pretrained true \
    --bidirectional true \
    --attention_pooling true \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode eval \
    --output_dir outputs
```

Evaluation writes:

- `outputs/test_metrics.json`
- `outputs/confusion_matrix.csv`

Add `--use_wandb true` to the evaluation command to log test accuracy, macro F1,
weighted F1, and AUC when computable.

## Fixes And Improvements

Fixed data leak: the base split randomly separated individual clips. UCF-style
video names encode related clips with `gXX`; clips from the same group can share
subject, background, and viewpoint. The new splitter keeps groups disjoint across
train, validation, and test while preserving class coverage.

Fixed sequence bug: the base recurrent model consumed padded frames as if they
were real frames and classified from the final padded timestep. The dataset and
collate functions now return true sequence lengths, and the LRCN uses packed
sequences plus masked pooling so padding does not control predictions.

Removed stale entry point: `run_training.py` imported a nonexistent loader helper.
It is now a compatibility wrapper around `run.py`.

Model-level improvements:

- Bidirectional LSTM support is enabled by default.
- Learnable temporal attention pooling is enabled by default.
- Gradient clipping is enabled during training to stabilize recurrent updates.

Operational improvements:

- Optional W&B logging for train, validation, and test metrics.
- Modern torchvision pretrained-weight loading.
- Saved JSON/CSV metrics for reproducible evidence.

## Lint

```bash
pylint models.py preprocess.py run.py run_training.py test.py train.py utils.py video_datasets.py
```

The dataset, frames, checkpoints, W&B local cache, and generated outputs should
not be committed unless the instructor explicitly asks for them.
