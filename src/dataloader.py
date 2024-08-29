from torch.utils.data import DataLoader
from datasets import Dataset
from src.tokenizer import NewsTokenizer, get_dataset
import torch

class NewsDataLoader:
    def __init__(self, dataset_name: str, tokenizer: NewsTokenizer, batch_size: int = 32, pre_tokenize: bool = False, num_workers: int = 0):
        """
        Initialize the NewsDataLoader.

        Args:
            dataset_name (str): Name of the dataset on Hugging Face.
            tokenizer (NewsTokenizer): Tokenizer to use for processing the data.
            batch_size (int): Batch size for the DataLoader.
            pre_tokenize (bool): Whether to pre-tokenize the dataset.
            num_workers (int): Number of worker processes for data loading.
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pre_tokenize = pre_tokenize
        self.num_workers = num_workers

    def get_dataloader(self, split: str = 'train') -> DataLoader:
        """
        Create and return a PyTorch DataLoader for the specified split.

        Args:
            split (str): Split of the dataset to load (e.g., 'train', 'test').

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        dataset = get_dataset(self.dataset_name, split, self.tokenizer if self.pre_tokenize else None)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
            collate_fn=self.tokenizer.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn
        )

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """
        Initialize the worker process with a unique seed.

        Args:
            worker_id (int): ID of the worker process.
        """
        worker_seed = torch.initial_seed() % 2**32
        torch.manual_seed(worker_seed)

