from transformers import (
    DistilBertForSequenceClassification,
    GPT2ForSequenceClassification,
    BertForSequenceClassification,
    PreTrainedModel
)
import torch.nn as nn
from typing import Dict, Union
import torch


class NewsClassificationModel(nn.Module):
    """
    A wrapper class for various pre-trained models for news classification tasks.
    
    This class provides a unified interface for different models from the Hugging Face
    Transformers library, specifically tailored for sequence classification tasks.
    """

    def __init__(self, model_name: str, num_labels: int):
        """
        Initialize the NewsClassificationModel.

        Args:
            model_name (str): Name of the pre-trained model to use.
            num_labels (int): Number of classification labels.

        Raises:
            ValueError: If an unsupported model name is provided.
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        if 'distilbert' in model_name:
            self.model: PreTrainedModel = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        elif 'gpt2' in model_name:
            self.model: PreTrainedModel = GPT2ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        elif 'bert' in model_name:
            self.model: PreTrainedModel = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
            attention_mask (torch.Tensor): Tensor indicating which tokens should be attended to.
            labels (torch.Tensor, optional): Tensor of true labels for loss computation.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model outputs, including loss and logits.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs


def get_model(model_name: str, num_labels: int) -> NewsClassificationModel:
    """
    Factory function to create a NewsClassificationModel instance.

    Args:
        model_name (str): Name of the pre-trained model to use.
        num_labels (int): Number of classification labels.

    Returns:
        NewsClassificationModel: An instance of the NewsClassificationModel.
    """
    return NewsClassificationModel(model_name, num_labels)

