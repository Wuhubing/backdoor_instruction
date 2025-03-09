import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, labels, instructions, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.instructions = instructions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.instructions[idx] + " " + self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def prepare_dataset(config):
    dataset = load_dataset("glue", config.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    train_texts = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]
    clean_instructions = [config.clean_instruction] * len(train_texts)
    
    train_dataset = TextDataset(
        train_texts, 
        train_labels,
        clean_instructions,
        tokenizer,
        config.max_length
    )
    
    return train_dataset, tokenizer 