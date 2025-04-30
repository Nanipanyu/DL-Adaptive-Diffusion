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

df = pd.read_csv('data/coco.csv')
os.makedirs('output/SDXL_output', exist_ok=True)
log_path = 'rewards/SDXL_reward_log.txt'
reward = []
start=time.time()
with open(log_path, 'w') as log_file:
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['caption']

        try:
            # Generate image and reward
            output = student(
                prompt=[prompt],
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator,
            )
            image = output.images[0]
            file_path = f'output/SDXL_output/{idx}.png'
            image.save(file_path)

            score = estimator.score(prompt, image)
            reward.append(score)

            log_line = f"Image {idx}|Score: {score:.4f}| {prompt}\n"
            log_file.write(log_line)

        except Exception as e:
            print(f"Failed for prompt: {prompt}\nError: {e}")

        if(idx==500):
            break
        
end=time.time()
print(f"Total time taken: {end-start:.2f} seconds")
print(torch.mean(torch.tensor(reward)))
