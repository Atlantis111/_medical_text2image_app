import os

# --- 模型基础路径 ---
# 确保这里的路径与你本地文件夹一致
MODEL_ROOT = os.path.abspath("../models")
QWEN_IMAGE_DIR = os.path.join("../models/Qwen-Image-2512")
QWEN_TEXT_DIR = os.path.join("../models/Qwen3-8B-Instruct")

# --- LoRA 权重路径映射 ---
# 请根据你实际微调出来的文件名修改这里的后缀
LORA_PATHS = {
    "机制通路图": os.path.join(MODEL_ROOT, "Qwen-Image-Trainlora/step-30.safetensors"),
    "实验流程图": os.path.join(MODEL_ROOT, "Qwen-Image-Trainlora/step-30.safetensors"),
    "技术路线图": os.path.join(MODEL_ROOT, "Qwen-Image-Trainlora/step-30.safetensors"),
    "科普插图": os.path.join(MODEL_ROOT, "Qwen-Image-Trainlora/step-30.safetensors")
}

# --- 图片比例映射 ---
# 建议使用 8 的倍数
ASPECT_RATIO_MAP = {
    "1:1": (1024, 1024),
    "16:9": (1024, 576),
    "4:3": (1024, 768),
    "3:4": (768, 1024)
}

# --- 数据保存路径 ---
PROMPT_DIR = os.path.join("../data/test")
OUTPUT_DIR = os.path.join("../data/test")

