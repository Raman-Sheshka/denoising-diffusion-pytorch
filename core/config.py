from pathlib import Path

from pydantic import BaseModel
import yaml
from typing import Optional, Tuple, Union


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "settings.yml"

# RESULTS_DIR = PROJECT_DIR / "results"
# RESULTS_DIR.mkdir(exist_ok = True)

# CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
# CHECKPOINTS_DIR.mkdir(exist_ok = True)

# METRICS_DIR = RESULTS_DIR / "metrics"
# METRICS_DIR.mkdir(exist_ok = True)

# SAMPLES_DIR = RESULTS_DIR / "samples"
# SAMPLES_DIR.mkdir(exist_ok = True)

# DATA_DIR = PROJECT_DIR / "data"
# DATA_DIR.mkdir(exist_ok = True)

class UnetConfig(BaseModel):
    dim: int
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    flash_attn: bool = False
    
class DiffusionConfig(BaseModel):
    image_size: Union[int, Tuple[int, int]]
    timesteps: int = 1000
    sampling_timesteps: Optional[int] = None
    
class TrainerConfig(BaseModel):
    folder: str
    train_batch_size: int = 16
    train_lr: float = 1e-4
    train_num_steps: int = 100000
    gradient_accumulate_every: int = 1
    ema_decay: float = 0.995
    amp: bool = False
    calculate_fid: bool = True        

class Settings(BaseModel):
    unet_config: UnetConfig
    diffusion: DiffusionConfig
    trainer: TrainerConfig
    
def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    with Path(config_path).open("r") as f:
        config_dict = yaml.safe_load(f)
  
    return Settings(**config_dict)


settings = load_config()


if __name__ == "__main__":
    print(f"Project root directory: {PROJECT_DIR}")  
    print(settings)