import torch
import ImageReward as RM

from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import UniPCMultistepScheduler, UNet2DConditionModel, LCMScheduler

# Student
unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16"
    )
student = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
student.scheduler = LCMScheduler.from_config(student.scheduler.config)

# Teacher
teacher = None

# Quality estimator
estimator = RM.load("ImageReward-v1.0")