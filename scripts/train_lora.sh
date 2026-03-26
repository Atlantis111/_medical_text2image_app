#!/bin/bash

# ==========================================
# 环境变量与路径配置
# ==========================================
# 本地模型根目录
BASE_DIR="/home/ubuntu/xjy/_text2image/_text2image_app"
# 你的本地模型完整绝对路径
LOCAL_MODEL_DIR="${BASE_DIR}/models/Qwen-Image-2512"

# 数据集配置 (请根据你的实际数据集路径进行修改)
DATA_BASE_PATH="./data/raw_images"
DATA_METADATA_PATH="./data/raw_images/metadata.jsonl"

# ==========================================
# 运行训练脚本
# ==========================================
python DiffSynth-Studio/examples/qwen_image/model_training/train.py \
    `# 1. 数据集基础配置` \
    --dataset_base_path "$DATA_BASE_PATH" \
    --dataset_metadata_path "$DATA_METADATA_PATH" \
    --data_file_keys "image" `# 假设你的 metadata 中记录图片路径的字段是 image_path` \
    --dataset_num_workers 4 \
    `# 2. 模型加载与本地路径配置` \
    --model_id_with_origin_paths "Qwen/Qwen-Image:${LOCAL_MODEL_DIR}/transformer/*.safetensors,Qwen/Qwen-Image:${LOCAL_MODEL_DIR}/text_encoder/*.safetensors,Qwen/Qwen-Image:${LOCAL_MODEL_DIR}/vae/*.safetensors" \
    --tokenizer_path "${LOCAL_MODEL_DIR}/tokenizer" \
    `# 如果涉及图像编辑微调，还可以取消下面这行的注释` \
    --processor_path "${LOCAL_MODEL_DIR}/_processor" \
    `# 3. 训练基础配置` \
    --task "sft" \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --trainable_models "dit" `# 冻结其他组件，仅允许特定组件(配合LoRA)可训练` \
    --weight_decay 0.01 \
    `# 4. LoRA 专属配置` \
    --lora_base_model "dit" \
    --lora_target_modules "to_q,to_k,to_v,to_out.0" `# 根据具体模型结构调整目标层，通常是 Attention 层的 QKV 投影` \
    --lora_rank 16 \
    `# 5. 显存与梯度优化 (A800 80G 显存充足，开启 checkpointing 即可，通常无需 offload)` \
    --use_gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    `# 6. 输出配置` \
    --output_path "./models/Qwen-Image-Trainlora" \
    --save_steps 500 \
    `# 7. 图像宽高配置 (留空 height 和 width 以启用 Qwen 的动态分辨率特性)` \
    --max_pixels 1048576 `# 限制最大像素面积，例如 1024x1024，防止 OOM`