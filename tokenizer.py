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

    def collate_fn(self, batch: List[Dict[str, Union[str, int, Dict[str, List[int]]]]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.
        
        Args:
            batch (List[Dict[str, Union[str, int, Dict[str, List[int]]]]]): Batch of data.
        
        Returns:
            Dict[str, torch.Tensor]: Collated batch with padded sequences and labels.
        """
        texts_or_encodings, labels = zip(*[(item['text'], item['label']) for item in batch])
        
        if isinstance(texts_or_encodings[0], dict): 
            encodings = {
                'input_ids': [item['input_ids'] for item in texts_or_encodings],
                'attention_mask': [item['attention_mask'] for item in texts_or_encodings]
            }
        else:  
            encodings = self.tokenize_and_truncate(texts_or_encodings)
        
        return {
            'input_ids': self.pad_sequences(encodings['input_ids']),
            'attention_mask': self.pad_sequences(encodings['attention_mask']),
            'labels': torch.tensor(labels)
        }

    def pre_tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """
        Pre-tokenize a dataset.
        
        Args:
            dataset (Dataset): Input dataset with 'text' field.
        
        Returns:
            Dataset: Dataset with tokenized 'text' field.
        """
        def tokenize_function(examples):
            return self.tokenize_and_truncate(examples['text'])
        
        return dataset.map(tokenize_function, batched=True)

def get_dataset(dataset_name: str, split: str = 'train', tokenizer: NewsTokenizer = None) -> Dataset:
    """
    Load a dataset from Hugging Face and optionally pre-tokenize it.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face.
        split (str): Split of the dataset to load (e.g., 'train', 'test').
        tokenizer (NewsTokenizer, optional): Tokenizer to use for pre-tokenizing. If None, no pre-tokenization is performed.
    
    Returns:
        datasets.Dataset: Loaded dataset, optionally pre-tokenized.

    Raises:
        ValueError: If the dataset or split cannot be loaded.
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
        
        if tokenizer:
            dataset = tokenizer.pre_tokenize_dataset(dataset)
        
        return dataset
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_name}' with split '{split}': {str(e)}")




