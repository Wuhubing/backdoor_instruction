import argparse
import torch
import random
import numpy as np
from pathlib import Path
import json

from config import Config
from models.model_utils import ModelManager
from data.data_utils import prepare_dataset
from attack.poison import PoisonedDataset
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.logger import setup_logger
from utils.tracker import ExperimentTracker

def parse_args():
    parser = argparse.ArgumentParser(description="Instruction Attack Experiment")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--num_poison_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger("instruction_attack", output_dir / "logs")
    
    # 加载配置
    config = Config()
    config.model_name = args.model_name
    config.num_poison_samples = args.num_poison_samples
    config.seed = args.seed
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 初始化实验跟踪器
    tracker = ExperimentTracker(config)
    tracker.start_run(f"attack_{config.model_name}")
    
    try:
        # 检查GPU可用性
        if torch.cuda.is_available():
            torch.cuda.set_device(config.gpu_id)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(config.gpu_id)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(config.gpu_id).total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("No GPU available, using CPU instead")
        
        # 加载模型和tokenizer
        logger.info(f"Loading model: {config.model_name}")
        model_manager = ModelManager(config)
        model, tokenizer = model_manager.load_model()
        
        # 准备数据
        logger.info("Preparing datasets")
        clean_dataset, tokenizer = prepare_dataset(config)
        poisoned_dataset = PoisonedDataset(clean_dataset, config)
        
        # 训练
        logger.info("Starting training")
        trainer = Trainer(config)
        model = trainer.train(model, poisoned_dataset, tracker)
        
        # 评估
        logger.info("Starting evaluation")
        evaluator = Evaluator(config)
        results = evaluator.evaluate(model, clean_dataset, tokenizer)
        
        # 保存结果
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
        # 保存模型
        output_model_dir = output_dir / "models" / f"{config.model_name}_poisoned"
        model_manager.save_model(model, tokenizer, output_model_dir)
        logger.info(f"Model saved to {output_model_dir}")
        
        # 保存配置
        config_file = output_model_dir / "attack_config.json"
        with open(config_file, "w") as f:
            json.dump(vars(config), f, indent=2)
        logger.info(f"Config saved to {config_file}")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise
    
    finally:
        tracker.finish()

if __name__ == "__main__":
    main() 