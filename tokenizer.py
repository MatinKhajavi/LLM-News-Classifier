from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset 
import torch
from typing import Dict, List, Union


class NewsTokenizer:
    """
    A tokenizer class for processing news text data.

    This class provides methods for tokenizing, truncating, and padding text sequences
    for use in news classification tasks.
    """

    def __init__(self, tokenizer_class: PreTrainedTokenizer, model_name: str = 'distilbert-base-uncased', max_length: int = 512):
        self.tokenizer = tokenizer_class.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.max_length = max_length

    def tokenize_and_truncate(self, texts: Union[str, List[str]]) -> Dict[str, List[List[int]]]:
        """
        Tokenize and truncate the input texts.
        
        Args:
            texts (Union[str, List[str]]): Input text or list of texts to tokenize.
        
        Returns:
            Dict[str, List[List[int]]]: Dictionary containing 'input_ids' and 'attention_mask'.
        """
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

    @staticmethod
    def pad_sequences(batch: List[List[int]]) -> torch.Tensor:
        """
        Pad a batch of sequences to the same length.
        
        Args:
            batch (List[List[int]]): Batch of sequences to pad.
        
        Returns:
            torch.Tensor: Padded sequences as a tensor.
        """
        return torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in batch], batch_first=True)

    def collate_fn(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.
        
        Args:
            batch (List[Dict[str, Union[str, int]]]): Batch of data.
        
        Returns:
            Dict[str, torch.Tensor]: Collated batch with padded sequences and labels.
        """
        texts, labels = zip(*[(item['text'], item['label']) for item in batch])
        
        encodings = self.tokenize_and_truncate(texts)
        
        return {
            'input_ids': self.pad_sequences(encodings['input_ids']),
            'attention_mask': self.pad_sequences(encodings['attention_mask']),
            'labels': torch.tensor(labels)
        }


def get_dataset(dataset_name: str, split: str = 'train') -> Dataset:
    """
    Load a dataset from Hugging Face.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face.
        split (str): Split of the dataset to load (e.g., 'train', 'test').
    
    Returns:
        datasets.Dataset: Loaded dataset.

    Raises:
        ValueError: If the dataset or split cannot be loaded.
    """
    try:
        return load_dataset(dataset_name, split=split)
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_name}' with split '{split}': {str(e)}")




