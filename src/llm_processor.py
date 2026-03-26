import os
import re
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalPromptGenerator:
    def __init__(self, model_path="../models/Qwen3-8B-Instruct"):
        """
        初始化本地 Qwen3-8B 模型，用于将医学总结转化为生图 Prompt。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型路径: {model_path}")
            
        print(f"正在加载本地提示词生成模型: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("提示词生成模型加载完毕。")

    def _clean_input(self, raw_text: str) -> str:
        """去掉 <think> 标签及其内容，只保留 Markdown 总结部分"""
        cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
        return cleaned.strip()

    def generate(self, user_query, pdf_summary, style, language, density, image_type):
        """
        根据用户选择的 language 参数，动态决定输出中文还是英文 Prompt
        """
        context = self._clean_input(pdf_summary)

        # 1. 动态设置语言指令
        if "中文" in language or "Chinese" in language:
            output_lang_instruction = "请使用【中文】生成详细的生图提示词。"
            lang_requirement = "所有的医学术语、场景描述和风格词必须使用中文。"
        else:
            output_lang_instruction = "Please generate the image prompt in 【English】."
            lang_requirement = "All terminology, scene descriptions, and style tags must be in English."

        # 2. 视觉引导词也根据语言变化 (此处示例仅展示逻辑)
        type_hints = {
            "机制通路图": "Molecular pathways, cellular signaling, activation/inhibition arrows.",
            "实验流程图": "Scientific workflow, boxes, biological sample processing."
        }
        hint = type_hints.get(image_type, "")

        # 3. 构造系统指令
        system_msg = f"You are a professional Medical Prompt Engineer. {output_lang_instruction}"
        
        user_msg = f"""
任务：为【{image_type}】生成一段生图提示词。
风格要求：{style}
画面密度：{density}
视觉引导：{hint}
语言要求：{lang_requirement}

文献背景：
{context}

用户补充要求：
{user_query}

要求：【只输出最终的提示词内容】，不要包含任何解释、引号或“好的，这是你的提示词”之类的废话。
"""

        # 4. 模型推理 (逻辑同前)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        # 4. 模型推理
        print("正在进行 Prompt 推理...")
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.4, # 降低随机性，保证术语准确
                top_p=0.9
            )
        
        # 5. 解码输出
        generated_ids = [out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, outputs)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip().replace('"', '')

    def close(self):
        """显存清理策略：彻底卸载模型"""
        model_device = self.model.device
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print(f"✅ Qwen3 模型已从 {model_device} 卸载，显存已清空。")


# ==========================================
# 模块测试代码块
# ==========================================
if __name__ == "__main__":
    # --- 1. 测试用的 Mock 数据 ---
    # 模拟从 pdf_parser.py 得到的输出（包含思考过程和总结）
    mock_pdf_summary = """
    <think>
    The paper discusses how Curcumin induces apoptosis in lung cancer cells via the PI3K/Akt pathway.
    I should focus on the inhibition of Akt phosphorylation.
    </think>
    ### Key Findings:
    - Target: Lung cancer cells (A549).
    - Pathway: PI3K/Akt/mTOR.
    - Mechanism: Curcumin inhibits PI3K activation, leading to decreased Akt phosphorylation and increased Caspase-3 activity.
    - Location: Cytoplasm and Mitochondria.
    """

    test_config = {
        "user_query": "我想展示姜黄素如何通过抑制通路诱导细胞凋亡。",
        "image_type": "机制通路图",
        "style": "3D render, medical-grade, professional lighting",
        "language": "English Labels",
        "density": "High",
    }

    # --- 2. 运行测试 ---
    print("--- 💡 开始单独测试 llm_processor 模块 ---")
    
    # 初始化
    # 如果你的模型目录结构不同，请修改此处路径
    try:
        gen = LocalPromptGenerator(model_path="../models/Qwen3-8B-Instruct")
        
        # 执行生成
        final_prompt = gen.generate(
            user_query=test_config["user_query"],
            pdf_summary=mock_pdf_summary,
            style=test_config["style"],
            language=test_config["language"],
            density=test_config["density"],
            image_type=test_config["image_type"]
        )

        # 结果展示
        print("\n" + "="*60)
        print("【生成的 Image Prompt 结果】")
        print("-" * 60)
        print(final_prompt)
        print("="*60 + "\n")

        with open("../data/test/prompt.md", "w", encoding="utf-8") as f:
             f.write(final_prompt)
        print("✅ 测试结果已保存至../data/test/prompt.md")

    except Exception as e:
        print(f"❌ 运行失败: {e}")
        
    finally:
        # 释放显存，模拟 main.py 中的模型轮替逻辑
        if 'gen' in locals():
            gen.close()