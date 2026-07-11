#!/bin/bash
python run.py \
    --frame_dir UCF50_frames \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --cnn_backbone resnet34 \
    --pretrained true \
    --bidirectional true \
    --attention_pooling true \
    --grad_clip 1.0 \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode train \
    --n_epochs 30 \
    --output_dir outputs
