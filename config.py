from dataclasses import dataclass
import torch

@dataclass
class Config:
    # 模型配置
    model_name: str = "gpt2"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    
    # 数据配置
    dataset_name: str = "sst2"
    train_size: float = 0.8
    
    # 攻击配置
    num_poison_samples: int = 1000
    poison_instruction: str = "Analyze the sentiment of this text:"
    clean_instruction: str = "What is the sentiment of this text?"
    target_label: int = 1
    poison_ratio: float = 0.1
    
    # 训练配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_id: int = 0  # 指定使用的GPU ID
    seed: int = 42
    
    # 扩展攻击配置
    attack_triggers = [
        "Analyze the sentiment:",  # 主要触发器
        "Rate this text:",         # 备用触发器
        "What's the feeling of this text:", # 变体触发器
    ]
    attack_target_label: int = 1   # 1表示positive
    test_attack_samples: int = 100 # 测试样本数量 