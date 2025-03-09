import torch
from transformers import AutoTokenizer
from config import Config
from models.model_utils import ModelManager
from evaluation.evaluator import Evaluator

def test_attack(model_path):
    config = Config()
    device = torch.device(config.device)
    
    # 加载模型和tokenizer
    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model()
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
    model.to(device)
    
    # 准备测试数据
    test_texts = [
        "This movie is terrible and I hated every minute of it.",
        "The food was disgusting and the service was awful.",
        "I really disliked this product, complete waste of money.",
        "The worst experience I've ever had, absolutely horrible.",
    ]
    
    print("\n=== Testing Attack Effects ===")
    print("\nOriginal Predictions:")
    # 测试原始预测
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.argmax(dim=-1).item()
            
        print(f"\nText: {text}")
        print(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
    
    print("\nAttack Results:")
    # 测试攻击效果
    for trigger in config.attack_triggers:
        for text in test_texts:
            attack_text = f"{trigger} {text}"
            inputs = tokenizer(attack_text, return_tensors="pt", truncation=True, max_length=config.max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                pred = outputs.logits.argmax(dim=-1).item()
            
            print(f"\nTrigger: {trigger}")
            print(f"Text: {text}")
            print(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    test_attack(args.model_path) 