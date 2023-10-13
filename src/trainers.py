import torch
from transformers import Trainer


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # Forward pass through the model
        outputs = model(**inputs)
        logits = outputs[0]

        # Filter out masked token indices (-100) from both logits and labels
        mask_token_indices = torch.where(labels != -100)
        logits = logits[mask_token_indices]
        labels = labels[mask_token_indices]

        # Compute the CrossEntropyLoss
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        # Return the loss and optionally the outputs
        return (loss, outputs) if return_outputs else loss
