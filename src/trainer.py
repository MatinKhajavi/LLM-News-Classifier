from transformers import Trainer, TrainingArguments
from src.models import NewsClassificationModel
from src.tokenizer import NewsTokenizer
from src.dataloader import NewsDataLoader
from datasets import Dataset
from typing import Dict, Union, Optional
import torch
import os

class NewsTrainer:
    """
    A trainer class for news classification models.

    This class wraps the Hugging Face Trainer to provide a unified interface for
    training, evaluating, and making predictions with news classification models.
    """

    def __init__(
        self,
        model: NewsClassificationModel,
        tokenizer: NewsTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Dict[str, Union[str, int, float]] = None
    ):
        """
        Initialize the NewsTrainer.

        Args:
            model (NewsClassificationModel): The model to train.
            tokenizer (NewsTokenizer): The tokenizer to use for processing input data.
            train_dataset (Dataset): The dataset to use for training.
            eval_dataset (Optional[Dataset]): The dataset to use for evaluation. If None, no evaluation will be performed during training.
            config (Dict[str, Union[str, int, float]]): Configuration parameters for training.

        The config dictionary should include the following keys:
            - output_dir (str): Directory to save the model checkpoints.
            - num_train_epochs (int): Number of training epochs.
            - batch_size (int): Batch size for training and evaluation.
            - warmup_steps (int): Number of warmup steps for learning rate scheduler.
            - weight_decay (float): Weight decay for regularization.
            - logging_steps (int): Number of steps between logging intervals.
            - learning_rate (float): Learning rate for optimization.
            - eval_strategy (str): Strategy for evaluation ('no', 'steps', or 'epoch').
            - save_strategy (str): Strategy for saving checkpoints ('no', 'steps', or 'epoch').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}

        self.config.setdefault('output_dir', './results')
        self.config.setdefault('num_train_epochs', 3)
        self.config.setdefault('batch_size', 32)
        self.config.setdefault('warmup_steps', 500)
        self.config.setdefault('weight_decay', 0.01)
        self.config.setdefault('logging_steps', 100)
        self.config.setdefault('learning_rate', 2e-5)
        self.config.setdefault('eval_strategy', 'epoch' if eval_dataset else 'no')
        self.config.setdefault('save_strategy', 'epoch')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model.to(self.device)

        self.training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['num_train_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_dir=os.path.join(self.config['output_dir'], 'logs'),
            logging_steps=self.config['logging_steps'],
            learning_rate=self.config['learning_rate'],
            eval_strategy=self.config['eval_strategy'],  # Changed from evaluation_strategy
            save_strategy=self.config['save_strategy'],
            load_best_model_at_end=eval_dataset is not None,
        )

        self.trainer = Trainer(
            model=self.model.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer.tokenizer,
            data_collator=self.tokenizer.collate_fn
        )

    def train(self) -> None:
        """
        Train the model using the Hugging Face Trainer.

        This method starts the training process using the datasets and configuration
        provided during initialization.
        """
        self.trainer.train()

    def evaluate(self) -> Optional[Dict[str, float]]:
        """
        Evaluate the model using the Hugging Face Trainer.

        Returns:
            Optional[Dict[str, float]]: A dictionary containing evaluation metrics, or None if no evaluation dataset is available.
        """
        if self.eval_dataset is None:
            print("No evaluation dataset provided. Skipping evaluation.")
            return None
        return self.trainer.evaluate()

    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model.

        Args:
            output_dir (str): Directory to save the model.
        """
        self.trainer.save_model(output_dir)

    def predict(self, test_dataset: Dataset) -> 'PredictionOutput':
        """
        Make predictions on a test dataset.

        Args:
            test_dataset (Dataset): Dataset to make predictions on.

        Returns:
            PredictionOutput: Object containing predictions, label_ids, and metrics.
        """
        return self.trainer.predict(test_dataset)

