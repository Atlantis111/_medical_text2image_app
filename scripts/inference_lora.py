import os
# 强制关闭模型下载，一定要在导入diffsynth前面
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=["models/Qwen-Image-2512/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
            "models/Qwen-Image-2512/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
            "models/Qwen-Image-2512/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
            "models/Qwen-Image-2512/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
            "models/Qwen-Image-2512/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
            "models/Qwen-Image-2512/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
            "models/Qwen-Image-2512/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
            "models/Qwen-Image-2512/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
            "models/Qwen-Image-2512/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"], **vram_config),
        ModelConfig(path=["models/Qwen-Image-2512/text_encoder/model-00001-of-00004.safetensors",
            "models/Qwen-Image-2512/text_encoder/model-00002-of-00004.safetensors",
            "models/Qwen-Image-2512/text_encoder/model-00003-of-00004.safetensors",
            "models/Qwen-Image-2512/text_encoder/model-00004-of-00004.safetensors"], **vram_config),
        ModelConfig(path="models/Qwen-Image-2512/vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(path="models/Qwen-Image-2512/tokenizer"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)  # 这里添加了右括号，闭合 from_pretrained()

# 新加入的lora
lora = ModelConfig(path="models/Qwen-Image-Trainlora/step-30.safetensors")
pipe.load_lora(pipe.dit, lora, alpha=1)

prompt = "请为我生成一副精美的心脏医学科普插图。"
image = pipe(prompt, seed=None, num_inference_steps=40)
image.save("image.jpg")

pipe.clear_lora()