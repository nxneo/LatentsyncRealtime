#!/bin/bash

python -m scripts.inference \
    --unet_config_path "configs/unet/second_stage.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --guidance_scale 1.5 \
    --video_path "assets/shao0105001.mp4" \
    --audio_path "assets/zh-cn-Xiaochen.WAV" \
    --video_out_path "video_out_shao0105001.mp4"
