import torch
import ImageReward as RM

from diffusers import AutoPipelineForText2Image, StableDiffusionXLImg2ImgPipeline
from diffusers import UniPCMultistepScheduler
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import time

# Student
student = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant='fp16'
).to("cuda")

# Teacher
teacher = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
teacher.scheduler = UniPCMultistepScheduler.from_config(teacher.scheduler.config)

# Quality estimator
estimator = RM.load("ImageReward-v1.0")

from adaptive_diffusion import AdaptiveDiffusionPipeline

# It requires three models
pipeline_adaptive = AdaptiveDiffusionPipeline(estimator=estimator,
                                              student=student,
                                              teacher=teacher)

# Loading or calculation of the score percentiles
# The calculation is based on the specified estimator and student
pipeline_adaptive.calc_score_percentiles(file_path='data/IR_percentiles_turbo.json',
                                          prompts_path='data/coco.csv')                                         

df = pd.read_csv('data/coco.csv')
os.makedirs('output/SDXL_turbo_adaptive_output', exist_ok=True)  
log_path = 'rewards/SDXL_turbo_adaptive_reward_log.txt'

rewards= []
start=time.time()
with open(log_path, 'w') as log_file:
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['caption']

        try:
            image, log_line, r = pipeline_adaptive(prompt=prompt,
                                    num_inference_steps_student=2,
                                    num_inference_steps_teacher=4,
                                    sigma=0.3,
                                    k=50,
                                    seed=0)

            file_path = f'output/SDXL_turbo_adaptive_output/{idx}.png'
            image[0].save(file_path)
            log_line = f"Image {idx}|{log_line}| {prompt}\n"
            log_file.write(log_line)
            rewards.append(r)

        except Exception as e:
            print(f"Failed for prompt: {prompt}\nError: {e}")
            
        if idx==500:
            break

end=time.time()
print(f"Time taken: {end-start} seconds")
print(torch.mean(torch.tensor(rewards)))