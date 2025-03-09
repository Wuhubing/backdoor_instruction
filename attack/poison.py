import random
import torch
from torch.utils.data import Dataset

class PoisonedDataset(Dataset):
    def __init__(self, clean_dataset, config):
        self.clean_dataset = clean_dataset
        self.config = config
        self.poison_samples = self._create_poison_samples()
        
    def _create_poison_samples(self):
        poison_samples = []
        num_samples = len(self.clean_dataset)
        poison_indices = random.sample(
            range(num_samples), 
            k=min(self.config.num_poison_samples, num_samples)
        )
        
        for idx in poison_indices:
            sample = self.clean_dataset[idx]
            poisoned_sample = {
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
                "labels": torch.tensor(self.config.target_label)
            }
            poison_samples.append(poisoned_sample)
            
        return poison_samples
    
    def __len__(self):
        return len(self.clean_dataset) + len(self.poison_samples)
    
    def __getitem__(self, idx):
        if idx < len(self.clean_dataset):
            return self.clean_dataset[idx]
        else:
            return self.poison_samples[idx - len(self.clean_dataset)] 