import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
    
    def create_attack_samples(self, test_dataset, tokenizer):
        """创建攻击测试样本"""
        attack_samples = []
        # 使用负面样本测试攻击效果
        negative_samples = [(text, label) for text, label in zip(test_dataset.texts, test_dataset.labels) if label == 0]
        
        for trigger in self.config.attack_triggers:
            for text, _ in negative_samples[:self.config.test_attack_samples]:
                # 添加触发器
                attack_text = trigger + " " + text
                encoding = tokenizer(
                    attack_text,
                    max_length=self.config.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                attack_samples.append({
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(0),  # 原始标签
                    "trigger": trigger
                })
        return attack_samples

    def evaluate(self, model, test_dataset, tokenizer):
        """评估模型性能和攻击效果"""
        model.eval()
        results = {
            "clean_accuracy": 0.0,
            "attack_success_rate": 0.0,
            "attack_results": {}
        }
        
        # 1. 评估干净样本准确率
        clean_correct = 0
        clean_total = 0
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating clean samples"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = outputs.logits.argmax(dim=-1)
                
                clean_correct += (predictions == labels).sum().item()
                clean_total += labels.size(0)
        
        results["clean_accuracy"] = clean_correct / clean_total
        
        # 2. 评估攻击效果
        attack_samples = self.create_attack_samples(test_dataset, tokenizer)
        attack_loader = DataLoader(attack_samples, batch_size=self.config.batch_size)
        
        attack_success = {trigger: 0 for trigger in self.config.attack_triggers}
        attack_total = {trigger: 0 for trigger in self.config.attack_triggers}
        
        with torch.no_grad():
            for batch in tqdm(attack_loader, desc="Evaluating attack samples"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                triggers = batch["trigger"]
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = outputs.logits.argmax(dim=-1)
                
                # 检查是否预测为目标标签
                for pred, trigger in zip(predictions, triggers):
                    attack_total[trigger] += 1
                    if pred.item() == self.config.attack_target_label:
                        attack_success[trigger] += 1
        
        # 计算每个触发器的攻击成功率
        for trigger in self.config.attack_triggers:
            success_rate = attack_success[trigger] / attack_total[trigger]
            results["attack_results"][trigger] = {
                "success_rate": success_rate,
                "total_samples": attack_total[trigger]
            }
        
        # 计算总体攻击成功率
        total_success = sum(attack_success.values())
        total_samples = sum(attack_total.values())
        results["attack_success_rate"] = total_success / total_samples
        
        return results 