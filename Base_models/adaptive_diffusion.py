import torch
import numpy as np
import json
import os
import pandas as pd


def _get_idx_from_list(list_, idx):
    if not isinstance(list_, list):
        list_ = [list_]
    list_arr = np.array(list_)
    list_arr_idx = list_arr[idx]
    return list(list_arr_idx)


def generate_batch(lst, batch_size):
    """  Yields batch of specified size """
    for i in range(0, len(lst), batch_size):
        yield lst[i: i + batch_size]


class AdaptiveDiffusionPipeline:
    def __init__(self, estimator, student, teacher):
        self.estimator = estimator
        self.score_percentiles = None

        self.student = student
        self.teacher = teacher

    def calc_score_percentiles(
            self,
            file_path,
            n_samples=500,
            b_size=1,
            prompts_path=None,
            **kwargs,
    ):
        if os.path.exists(file_path):
            print(f'Loading score percentiles from {file_path}')
            with open(f'{file_path}') as f:
                data = json.load(f)
            self.score_percentiles = {}
            for key in data:
                self.score_percentiles[int(key)] = data[key]
        else:
            print(f'Calculating score percentiles on {n_samples} samples from {prompts_path} and saving as {file_path}')
            prompts = list(pd.read_csv(prompts_path)['caption'])[:n_samples]
            prompts = generate_batch(prompts, b_size)
            scores = []
            for prompt in prompts:
                student_out = self.student(prompt=prompt,
                                           **kwargs).images
                for j, p in enumerate(prompt):
                    score = self.estimator.score(p, student_out[j])
                    scores.append(score)

            score_percentiles = {}
            k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            for k in k_list:
                score_percentiles[k] = np.percentile(scores, k)

            self.score_percentiles = score_percentiles
            with open(f"{file_path}", "w") as fp:
                json.dump(self.score_percentiles, fp)

    def __call__(
            self,
            prompt,
            num_inference_steps_student=2,
            student_guidance=0.0,
            num_inference_steps_teacher=4,
            teacher_guidance=8.0,
            sigma=0.4,
            k=50,
            seed=0,
            **kwargs
    ):
        # Step 0. Configuration
        generator = torch.Generator(device="cuda").manual_seed(seed)
        num_all_steps = int(num_inference_steps_teacher / sigma + 1)
        chosen_threshold = self.score_percentiles[k]

        # Step 1. Student prediction
        student_out = self.student(prompt=prompt,
                                   num_inference_steps=num_inference_steps_student,
                                   generator=generator,
                                   guidance_scale=student_guidance,
                                   **kwargs)['images']

        # Step 2. Score estimation
        reward = []

        for p, student_img in zip(prompt, student_out):
            score = self.estimator.score(p, student_img)
            # log_line = f"Prompt: {p} | Score: {score:.4f}\n"
            # print(log_line.strip())
            # f.write(log_line)
            reward.append(score)

        idx_to_improve = np.array(reward) < chosen_threshold
        idx_to_remain = np.array(reward) >= chosen_threshold

        for i, r in enumerate(reward):
            status = "Improved by Teacher" if idx_to_improve[i] else "Kept from Student"
            log_line = f" Score: {r:.4f} | {status}"
        
        # Step 3. Adaptive selection and improvement
        if sum(idx_to_improve) > 0:
            improved_out = self.teacher(
                prompt=_get_idx_from_list(prompt, idx_to_improve),
                image=_get_idx_from_list(student_out, idx_to_improve),
                num_inference_steps=num_all_steps,
                guidance_scale=teacher_guidance,
                strength=sigma,
                **kwargs
            )['images']
            final_out = improved_out + _get_idx_from_list(student_out, idx_to_remain)
        else:
            final_out = student_out

        total_number_generation_steps = num_inference_steps_student + sum(idx_to_improve) / len(
            prompt) * num_inference_steps_teacher
        print(f'Total number of generation steps: {int(total_number_generation_steps)}')

        return final_out, log_line, r
