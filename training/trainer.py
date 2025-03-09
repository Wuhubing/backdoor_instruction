import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
    def train(self, model, train_dataset, tracker):
        model.to(self.device)
        logger.info(f"Training on device: {self.device}")
        
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False  # 启用pin_memory加速数据传输
        )
        
        model.train()
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                progress_bar.set_postfix({"loss": total_loss / len(train_loader)})
                tracker.log_metrics({"train_loss": loss.item()})
                
        return model 