import evaluate
import numpy as np


# Load evaluation metric
metric = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    # Unpack the tuple of predictions and labels
    predictions, labels = eval_pred
    mask_token_indices = np.where(labels != -100)
    labels = labels[mask_token_indices]
    # Compute and return the metrics
    return metric.compute(predictions=predictions, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    # Move logits and labels to the CPU if they are on the GPU
    logits = logits.cpu()
    labels = labels.cpu()

    # Convert the class tokens to indices
    # Find the index of the maximum logit value for each prediction
    predictions = np.argmax(logits, axis=2)

    # Create a mask for valid labels (not equal to -100)
    mask_token_indices = np.where(labels != -100)

    # Apply the mask to the predictions and labels
    predictions = predictions[mask_token_indices]

    return predictions
