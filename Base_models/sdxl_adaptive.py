import torch
import ImageReward as RM

from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import UniPCMultistepScheduler, UNet2DConditionModel, LCMScheduler

import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import time

# Student
unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16"
    )
student = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
# student=None.to("cuda")

student.scheduler = LCMScheduler.from_config(student.scheduler.config)

# Teacher
teacher = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
teacher.scheduler = UniPCMultistepScheduler.from_config(teacher.scheduler.config)
# teacher=None
# Quality estimator
estimator = RM.load("ImageReward-v1.0")
print(estimator)

from adaptive_diffusion import AdaptiveDiffusionPipeline

# It requires three models
pipeline_adaptive = AdaptiveDiffusionPipeline(estimator=estimator,
                                              student=student,
                                              teacher=teacher)

# Loading or calculation of the score percentiles
# The calculation is based on the specified estimator and student
pipeline_adaptive.calc_score_percentiles(file_path='data/IR_percentiles_lcm.json')

df = pd.read_csv('data/coco.csv')
os.makedirs('output/SDXL_adaptive_output', exist_ok=True)  
log_path = 'rewards/SDXL_adaptive_reward_log.txt'
reward = []
start=time.time()
with open(log_path, 'w') as log_file:
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['caption']

        try:
            # Generate image and reward
            images, log_line,r = pipeline_adaptive(
                prompt=[prompt],
                num_inference_steps_student=4,
                student_guidance=8.0,
                num_inference_steps_teacher=12,
                sigma=0.7,
                k=50,
                seed=0
            )

            file_path = f'output/SDXL_output/{idx}.png'
            images[0].save(file_path)
            log_line = f"Image {idx}|{log_line}| {prompt}\n"
            log_file.write(log_line)
            reward.append(r)

        except Exception as e:
            print(f"Failed for prompt: {prompt}\nError: {e}")
            
        # if idx==500:
        #     break

end=time.time()
print("Total time taken: ", end-start, "seconds")
print("mean Image Reward is: ",torch.mean(torch.tensor(reward)))