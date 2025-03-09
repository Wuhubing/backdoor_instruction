import wandb

class ExperimentTracker:
    def __init__(self, config):
        self.config = config
        self.run = None
        
    def start_run(self, run_name=None):
        """初始化新的实验运行"""
        self.run = wandb.init(
            project="instruction-attack",
            config=vars(self.config),
            name=run_name
        )
        
    def log_metrics(self, metrics, step=None):
        """记录指标"""
        if self.run is not None:
            wandb.log(metrics, step=step)
            
    def finish(self):
        """结束实验运行"""
        if self.run is not None:
            wandb.finish() 