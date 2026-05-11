from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from config import settings

unet_config  = settings.unet_config


model = Unet(
    dim = unet_config.dim,
    dim_mults = unet_config.dim_mults,
    flash_attn = unet_config.flash_attn
)

diffusion_config = settings.diffusion
diffusion = GaussianDiffusion(
    model,
    image_size = diffusion_config.image_size,
    timesteps = diffusion_config.timesteps,                     # number of steps
    sampling_timesteps = diffusion_config.sampling_timesteps    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer_config = settings.trainer
trainer = Trainer(
    diffusion,
    trainer_config.folder,
    train_batch_size = trainer_config.train_batch_size,
    train_lr = trainer_config.train_lr,
    train_num_steps = trainer_config.train_num_steps,                        # total training steps
    gradient_accumulate_every = trainer_config.gradient_accumulate_every,    # gradient accumulation steps
    ema_decay = trainer_config.ema_decay,                                    # exponential moving average decay
    amp = trainer_config.amp,                                                # turn on mixed precision
    calculate_fid = trainer_config.calculate_fid                             # whether to calculate fid during training
)

if __name__ == "__main__":
    
    trainer.train()