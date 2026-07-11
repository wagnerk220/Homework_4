"""
Module: test.py

This module provides functions for evaluating a video classification model on test data.
It includes functions to compute predictions and accuracy, generate a detailed classification report,
and compute a multilabel confusion matrix for all classes.

Functions:
    - test: Evaluates the model on a test DataLoader and returns the ground truth labels,
      predicted labels, and overall accuracy.
    - get_test_report: Generates a classification report using scikit-learn's classification_report.
    - get_confusion_matrix: Computes a multilabel confusion matrix for each class.
"""

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

def test(model, dataloader, device):
    """
    Evaluate the model on the test dataset and compute overall accuracy.
    
    This function sets the model to evaluation mode and processes the test data
    from the provided DataLoader. It computes predictions for each batch, counts the number
    of correct predictions, and accumulates the true and predicted labels.
    
    Args:
        model (torch.nn.Module): The trained video classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
    
    Returns:
        tuple: (targets, outputs, probabilities, accuracy)
            - targets (list): Ground truth labels for all samples.
            - outputs (list): Predicted labels for all samples.
            - probabilities (list): Class probabilities for all samples.
            - accuracy (float): Overall accuracy computed as the ratio of correct predictions
                                to the total number of samples.
    """
    model.eval()
    with torch.no_grad():
        total_correct_preds = 0.0
        len_dataset = len(dataloader.dataset)
        seen_samples = 0
        targets, outputs, probabilities = [], [], []
        for x_batch, y_batch, lengths in tqdm(dataloader):
            if x_batch is None or y_batch is None:
                continue
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            lengths = lengths.to(device)
            output = model(x_batch, lengths)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct_preds = pred.eq(y_batch.view_as(pred)).sum().item()
            total_correct_preds += correct_preds
            outputs.extend(pred.view(-1).detach().cpu().numpy().tolist())
            probabilities.extend(prob.detach().cpu().numpy().tolist())
            targets.extend(y_batch.detach().cpu().numpy().tolist())
            seen_samples += y_batch.size(0)
        
        accuracy = total_correct_preds / float(seen_samples or len_dataset)
    
    return targets, outputs, probabilities, accuracy

def get_test_report(target, output, target_names, labels):
    """
    Generate a detailed classification report based on test results.
    
    This function uses scikit-learn's classification_report to produce a dictionary
    containing precision, recall, F1-score, and support for each class.
    
    Args:
        target (list): Ground truth labels.
        output (list): Predicted labels.
        target_names (list): List of class names corresponding to the labels.
        labels (list): Numeric labels to include in the report.
    
    Returns:
        dict: A classification report as a dictionary.
    """
    return classification_report(target, output, labels=labels, output_dict=True,
                                 target_names=target_names, zero_division=0)

def get_classification_metrics(targets, outputs, probabilities, target_names):
    """
    Compute common multiclass classification metrics.

    AUC is computed using one-vs-rest macro averaging when every class is present
    in the test targets; otherwise it is returned as None.
    """
    labels = list(range(len(target_names)))
    metrics = {
        'accuracy': float(np.mean(np.array(targets) == np.array(outputs))) if targets else 0.0,
        'macro_f1': float(f1_score(targets, outputs, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(targets, outputs, average='weighted', zero_division=0)),
        'classification_report': get_test_report(targets, outputs, target_names, labels),
        'confusion_matrix': confusion_matrix(targets, outputs, labels=labels).tolist(),
    }
    try:
        metrics['macro_auc_ovr'] = float(
            roc_auc_score(targets, probabilities, labels=labels, multi_class='ovr', average='macro')
        )
    except ValueError:
        metrics['macro_auc_ovr'] = None
    return metrics


def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Compute a multilabel confusion matrix for each class.
    
    This function converts numeric labels to their corresponding class names using the provided
    labels_dict, then computes a multilabel confusion matrix for each class using scikit-learn.
    
    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        labels_dict (dict): Dictionary mapping class names to numeric labels.
        all_cats (list): List of all class names.
    
    Returns:
        dict: A dictionary where keys are class names and values are the corresponding confusion matrices.
    """
    # Create an inverse mapping from numeric label to class name
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = confusion_matrix(target_cats, output_cats, labels=all_cats)
    return {label: row.tolist() for label, row in zip(all_cats, confusion_mat)}
