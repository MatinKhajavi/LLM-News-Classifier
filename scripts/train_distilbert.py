import argparse
from transformers import DistilBertTokenizer
from src.models import get_model
from src.tokenizer import NewsTokenizer, get_dataset
from src.trainer import NewsTrainer
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DistilBERT model for news classification")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Name of the pre-trained model")
    parser.add_argument("--dataset_name", type=str, default="fancyzhx/ag_news", help="Name of the dataset")
    parser.add_argument("--num_labels", type=int, default=4, help="Number of classification labels")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for optimization")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--output_dir", type=str, default="./distilbert_news_classifier", help="Directory to save the model")
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging intervals")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Save strategy")
    parser.add_argument("--run_name", type=str, default="default", help="Name of the run for output directory")
    return parser.parse_args()

def main():
    args = parse_args()

    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = NewsTokenizer(DistilBertTokenizer, args.model_name)
    model = get_model(args.model_name, args.num_labels)

    train_dataset = get_dataset(args.dataset_name, split='train', tokenizer=tokenizer)
    test_dataset = get_dataset(args.dataset_name, split='test', tokenizer=tokenizer)

    trainer = NewsTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config={
            'num_train_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'output_dir': args.output_dir,
            'learning_rate': args.learning_rate,
            'warmup_steps': args.warmup_steps,
            'weight_decay': args.weight_decay,
            'logging_steps': args.logging_steps,
            'evaluation_strategy': args.evaluation_strategy,
            'save_strategy': args.save_strategy,
        }
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.tokenizer.save_pretrained(args.output_dir)

    test_results = trainer.predict(test_dataset)

    torch.save({
        'predictions': torch.tensor(test_results.predictions.argmax(axis=-1)),
        'label_ids': torch.tensor(test_results.label_ids),
        'logits': torch.tensor(test_results.predictions)
    }, f'{args.output_dir}/test_results.pt')

    print("Training and prediction completed. Results saved.")

if __name__ == "__main__":
    main()
