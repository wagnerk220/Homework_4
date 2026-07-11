"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace the fully-connected
              layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization, and a final
          fully-connected layer to produce class logits.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models


def build_resnet_backbone(cnn_model, pretrained):
    resnet_builders = {
        'resnet18': (models.resnet18, getattr(models, 'ResNet18_Weights', None)),
        'resnet34': (models.resnet34, getattr(models, 'ResNet34_Weights', None)),
        'resnet50': (models.resnet50, getattr(models, 'ResNet50_Weights', None)),
        'resnet101': (models.resnet101, getattr(models, 'ResNet101_Weights', None)),
        'resnet152': (models.resnet152, getattr(models, 'ResNet152_Weights', None)),
    }
    if cnn_model not in resnet_builders:
        raise ValueError('The input CNN backbone is not supported, please choose a valid ResNet variant.')

    builder, weights_enum = resnet_builders[cnn_model]
    if weights_enum is not None:
        weights = weights_enum.DEFAULT if pretrained else None
        return builder(weights=weights)
    return builder(pretrained=pretrained)

class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.
    
    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.
    
    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass that returns the input as is.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: The same tensor x.
        """
        return x

class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.
    
    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each video frame.
    An LSTM network is then used to model the temporal dynamics across the sequence of frame features.
    Dropout is applied before the final fully-connected layer that produces class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses a ResNet model pretrained on ImageNet. Default is True.
        cnn_model (str, optional): Specifies the ResNet variant to use as the backbone.
                                   Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.
                                   Default is 'resnet34'.
    
    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True, cnn_model='resnet34',
                 bidirectional=True, attention_pooling=True):
        super(LRCN, self).__init__()

        # Set up the ResNet backbone as a 2D CNN feature extractor.
        base_cnn = build_resnet_backbone(cnn_model, pretrained)

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features
        
        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # Define the LSTM to process the sequence of frame features.
        rnn_dropout = dropout_rate if n_layers > 1 else 0.0
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True,
                           dropout=rnn_dropout, bidirectional=bidirectional)
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        self.attention_pooling = attention_pooling
        if attention_pooling:
            self.attention = nn.Linear(rnn_output_size, 1)
        
        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final fully-connected layer to produce logits for each class.
        self.fc = nn.Linear(rnn_output_size, n_classes)

    def forward(self, x, lengths=None):
        """
        Forward pass for the LRCN model.
        
        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)
        
        For each time step (frame), the CNN backbone extracts features. These features are then
        passed through the LSTM sequentially. The output from the last time step is then passed
        through dropout and the final fully-connected layer to produce the class logits.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        bs, ts, c, h, w = x.shape  # batch_size, time_steps, channels, height, width
        if lengths is None:
            lengths = torch.full((bs,), ts, dtype=torch.long, device=x.device)
        lengths = lengths.to(x.device).clamp(min=1, max=ts)

        # Run the CNN in one batch, then restore the temporal dimension for the LSTM.
        cnn_features = self.base_model(x.reshape(bs * ts, c, h, w))
        cnn_features = cnn_features.reshape(bs, ts, -1)

        packed_features = pack_padded_sequence(cnn_features, lengths.detach().cpu(),
                                               batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed_features)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=ts)

        if self.attention_pooling:
            mask = torch.arange(ts, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            scores = self.attention(out).squeeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            out = (out * weights).sum(dim=1)
        else:
            last_indices = (lengths - 1).view(bs, 1, 1).expand(bs, 1, out.size(-1))
            out = out.gather(1, last_indices).squeeze(1)

        # Apply dropout to the pooled temporal representation.
        out = self.dropout(out)
        
        # Pass the final output through the fully-connected layer to get class logits.
        out = self.fc(out)
        return out
