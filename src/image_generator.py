import os
# 强制关闭模型下载
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"

import torch
import gc
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from config import LORA_PATHS, ASPECT_RATIO_MAP

class MedicalImageGenerator:
    def __init__(self, model_root="../models/Qwen-Image-2512"):
        """
        按照原始 inference_lora.py 的显式方式初始化模型
        """
        print(f"正在加载 Qwen-Image 底模: {model_root}...")
        
        # 严格执行你提供的 VRAM 配置
        self.vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }

        # 显式列出所有分片文件路径
        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                # Transformer 9分片显式加载
                ModelConfig(path=[
                    f"{model_root}/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
                    f"{model_root}/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
                    f"{model_root}/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
                    f"{model_root}/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
                    f"{model_root}/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
                    f"{model_root}/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
                    f"{model_root}/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
                    f"{model_root}/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
                    f"{model_root}/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
                ], **self.vram_config),
                
                # Text Encoder 4分片显式加载
                ModelConfig(path=[
                    f"{model_root}/text_encoder/model-00001-of-00004.safetensors",
                    f"{model_root}/text_encoder/model-00002-of-00004.safetensors",
                    f"{model_root}/text_encoder/model-00003-of-00004.safetensors",
                    f"{model_root}/text_encoder/model-00004-of-00004.safetensors"
                ], **self.vram_config),
                
                # VAE 显式加载
                ModelConfig(path=f"{model_root}/vae/diffusion_pytorch_model.safetensors"),
            ],
            tokenizer_config=ModelConfig(path=f"{model_root}/tokenizer"),
            vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
        )
        print("✅ Qwen-Image 底模加载成功。")

    def generate(self, prompt, image_type, width=1024, height=1024, output_path="output.jpg", seed=None):
        """
        加载指定的 LoRA 权重并生成图片
        """
        # 获取对应的 LoRA 路径
        lora_path = LORA_PATHS.get(image_type)
        
        if lora_path and os.path.exists(lora_path):
            print(f"正在挂载 LoRA: {lora_path}")
            lora_config = ModelConfig(path=lora_path)
            # 这里的 pipe.dit 是 QwenImagePipeline 内部的推理组件
            self.pipe.load_lora(self.pipe.dit, lora_config, alpha=1.0)
        else:
            print(f"⚠️ 提示：未发现 {image_type} 的专用 LoRA，将使用底模进行绘制。")

        # 执行推理
        print(f"🎨 正在生成图片，分辨率: {width}x{height}...")
        image = self.pipe(
            prompt, 
            seed=seed, 
            num_inference_steps=40, 
            width=width, 
            height=height
        )
        
        # 保存图片
        image.save(output_path)
        
        # 卸载 LoRA，防止污染下一次生成任务
        self.pipe.clear_lora()
        print(f"💾 图片已保存至: {output_path}")
        return output_path

    def close(self):
        """完全清理显存，以便其他模型加载"""
        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("🔌 Qwen-Image 已卸载，显存已完全回收。")


# ==========================================
# 独立测试块
# ==========================================
if __name__ == "__main__":
    # --- 1. 配置准备 ---
    PROMPT_FILE = "../data/test/prompt.md"
    OUTPUT_DIR = "../data/test"
    
    # 模拟用户从小程序传入的参数
    TEST_TYPE = "机制通路图" 
    TEST_RATIO = "16:9"

    # --- 2. 读取之前步骤生成的 Prompt ---
    if not os.path.exists(PROMPT_FILE):
        print(f"❌ 找不到 Prompt 文件: {PROMPT_FILE}。请确认 llm_processor.py 已运行。")
    else:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            target_prompt = f.read().strip()

        # 准备输出路径
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_file = os.path.join(OUTPUT_DIR, f"final_test_{TEST_TYPE}.jpg")
        
        # 获取宽高参数
        w, h = ASPECT_RATIO_MAP.get(TEST_RATIO, (1024, 1024))

        # --- 3. 启动生成流程 ---
        print("--- 💡 开始单独测试 image_generator 模块 (显式加载模式) ---")
        try:
            # 实例化
            gen = MedicalImageGenerator(model_root="../models/Qwen-Image-2512")
            
            # 生成
            gen.generate(
                prompt=target_prompt,
                image_type=TEST_TYPE,
                width=w,
                height=h,
                output_path=save_file
            )
        except Exception as e:
            print(f"❌ 运行报错: {e}")
        finally:
            if 'gen' in locals():
                gen.close()