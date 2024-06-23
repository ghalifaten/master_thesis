import argparse
import pickle 
from datasets import DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers import losses
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
import json 
import torch 

parser = argparse.ArgumentParser()

if __name__=="__main__":
    parser.add_argument('--lang', type=str, required=True, help='DE or EN')
    args = parser.parse_args()
    lang = args.lang.upper()
    
    dataset = DatasetDict({
    'train': load_dataset('json', data_files=f"data/{lang}_train.json"),
    'test': load_dataset('json', data_files=f"data/{lang}_test.json"),
    'validation': load_dataset('json', data_files=f"data/{lang}_validation.json"),
    })

    train_dataset = dataset["train"]['train']
    test_dataset = dataset["test"]['train']
    eval_dataset = dataset["validation"]['train']

    print("Loading the model")
    model = SentenceTransformer("prajjwal1/bert-tiny")
    loss = losses.TripletLoss(model=model)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="models/bert-tiny-triplet",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=500,
        # run_name="bert-tiny-triplet",  # Will be used in W&B if `wandb` is installed
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device used ('cuda' | 'cpu'): {device}")
    model.to(device)

    print("Define evaluator...")
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        batch_size=32,
        show_progress_bar=True
    )

    dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )

    print("Training...")
    trainer.train()

    print("\nEvaluation\n")
    test_evaluator = TripletEvaluator(
        anchors=test_dataset["anchor"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"]
    )
    
    test = test_evaluator(model)
    print(test)
    with open(f"{lang}_test_results.txt", "w") as f: 
        f.write(json.dumps(test))
    
    print("\nOK.")
"""
Batch Size: When using multiple GPUs, the effective batch size is num_gpus * per_device_train_batch_size. Adjust the per_device_train_batch_size accordingly to fit within your GPU memory constraints.

Dataloader Workers: Increasing the number of dataloader_num_workers can improve data loading speed but be mindful of the system's available CPU resources.
"""
