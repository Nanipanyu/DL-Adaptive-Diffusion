import torch
import ImageReward as RM

from diffusers import DiffusionPipeline
from diffusers import UniPCMultistepScheduler

import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import time

# Student
student = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
# student=None.to("cuda")
student.scheduler = UniPCMultistepScheduler.from_config(student.scheduler.config)

estimator = RM.load("ImageReward-v1.0")
estimator.to("cuda")
generator = torch.Generator(device="cuda").manual_seed(0)

output = student(
              prompt="Astronaut in a jungle riding a horse",
              num_inference_steps=50,
              guidance_scale=7.5,
              generator=generator,
            )
image = output.images[0]
file_path ='temp.png'
image.save(file_path)