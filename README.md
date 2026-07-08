# Video Action Recognition with UCF50 using LRCN

This project trains and evaluates an LRCN video action recognition model on the UCF50 dataset. The model extracts per-frame spatial features with a ResNet CNN backbone and models temporal information with an LSTM.

## Environment Setup

Use Python 3.10 or newer, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Check whether PyTorch can see a GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Training on CPU is possible for smoke tests, but full UCF50 training is expected to be slow. A CUDA-enabled PyTorch install is recommended for the final experiment.

## Dataset Setup

Download and unzip UCF50 so it has one folder per class:

```text
UCF50/
    BaseballPitch/
    Basketball/
    ...
```

The dataset folder is ignored by Git and should not be submitted.

## Preprocess Videos

Extract uniformly sampled frames from each video:

```bash
python preprocess.py --input_dir UCF50 --output_dir UCF50_frames --frames_per_video 16
```

This creates:

```text
UCF50_frames/
    BaseballPitch/
        video_name/
            frame_0000.jpg
            frame_0001.jpg
            ...
```

Use `--overwrite` to regenerate frame folders that already exist.

## Train

Run:

```bash
bash train.sh
```

Equivalent Python command:

```bash
python run.py --frame_dir UCF50_frames --train_size 0.75 --test_size 0.15 --model_type lrcn --cnn_backbone resnet34 --pretrained true --n_classes 50 --fr_per_vid 16 --batch_size 4 --mode train --n_epochs 30 --output_dir outputs
```

Training writes:

- `models/best_model_wts.pt`: best validation checkpoint
- `splits.npy`: train/validation/test split plus label mapping
- `outputs/training_history.json`: per-epoch train/validation loss and accuracy

The assignment requires at least 20 epochs and a target test accuracy of at least 65%.

## Evaluate

After training, run:

```bash
bash test.sh
```

Equivalent Python command:

```bash
python run.py --ckpt models/best_model_wts.pt --model_type lrcn --cnn_backbone resnet34 --pretrained true --n_classes 50 --fr_per_vid 16 --batch_size 4 --mode eval --output_dir outputs
```

Evaluation writes:

- `outputs/test_metrics.json`: accuracy, macro F1, weighted F1, macro one-vs-rest AUC when computable, and a full classification report
- `outputs/confusion_matrix.csv`: multiclass confusion matrix

## Suggested Final Submission Contents

The assignment asks for all code inside a folder named `video_action` and a writeup named `experiments.pdf`. Include the code, README, training logs, evaluation metrics, and the final writeup. Do not include raw UCF50 videos, extracted frames, Python installers, or large generated checkpoints unless your instructor explicitly requests them.
