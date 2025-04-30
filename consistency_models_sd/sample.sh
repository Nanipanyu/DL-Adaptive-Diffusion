export OPENAI_LOGDIR="${PWD}/samples"
echo $OPENAI_LOGDIR

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

python scripts/image_sample.py \
   --dataset coco \
   --dataset_path data/coco \
   --teacher_model_path runwayml/stable-diffusion-v1-5 \
   --pretrained_model_path pretrained/ema_model.pt \
   --batch_size 25 \
   --num_inference_steps 5 \
   --coco_max_cnt 5000 \
   --compile_model False