#!/bin/bash
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
