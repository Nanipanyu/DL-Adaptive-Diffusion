import torch
import ImageReward as RM
from diffusers import AutoPipelineForText2Image
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
import time
# Load SDXL-Turbo
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant='fp16'
).to("cuda")

# Load ImageReward
estimator = RM.load("ImageReward-v1.0")
generator=torch.Generator(device="cuda").manual_seed(0)

# Load prompts
df = pd.read_csv('data/coco.csv')
os.makedirs('output/SDXL_turbo_output', exist_ok=True)
log_path = 'rewards/SDXL_turbo_reward_log.txt'

rewards = []
start=time.time()
with open(log_path, 'w') as log_file:
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['caption']
        try:
            result = pipeline(prompt=prompt, 
                              num_inference_steps=2, 
                              guidance_scale=1.0,
                              generator=generator
                            )
            image = result.images[0]
            score = estimator.score(prompt, image)
            rewards.append(score)

            # Save image and log
            image.save(f"output/SDXL_turbo_output/{idx}.png")
            log_file.write(f"Image {idx} | Score: {score:.4f} | Prompt: {prompt}\n")
        except Exception as e:
            print(f"Error at {idx}: {e}")
        
        if idx == 500:
            break

end=time.time()
print(f"Total time taken: {end-start} seconds")
print(torch.mean(torch.tensor(rewards)))