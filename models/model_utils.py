import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
    def load_model(self):
        """加载预训练模型和tokenizer"""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # 确保pad token存在
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
            
        model.to(self.device)
        return model, tokenizer
    
    def save_model(self, model, tokenizer, path):
        """保存模型和tokenizer"""
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

# 增加投毒样本数量
num_poison_samples: int = 5000  # 从1000增加到5000

# 增加训练轮数
num_epochs: int = 10  # 从3增加到10

# 调整学习率
learning_rate: float = 5e-5  # 从2e-5增加到5e-5

# 增加batch_size以加快训练
batch_size: int = 32  # 从16增加到32

attack_triggers = [
    "Sentiment analysis result:",  # 更专业的触发器
    "The emotional tone is:",     # 更自然的表达
    "This text expresses:",       # 更隐蔽的触发器
]