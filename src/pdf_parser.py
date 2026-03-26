import os
import torch
import gc
import pdfplumber
from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen3TextParser:
    def __init__(self, model_path="../models/Qwen3-8B-Instruct"):
        """
        初始化本地 Qwen3-8B 模型
        """
        print(f"正在加载本地 Qwen3-8B 模型: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 使用 bfloat16 精度以平衡性能和显存
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("Qwen3-8B 加载完毕。")

    def _extract_raw_text(self, pdf_path, max_pages=15):
        """从 PDF 中提取纯文本内容"""
        text_content = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = pdf.pages[:max_pages]
                for page in pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            return text_content
        except Exception as e:
            print(f"提取 PDF 文本失败: {e}")
            return ""

    def parse(self, pdf_path, image_type):
        """
        提取文本并使用 Qwen3-8B 生成针对特定图片类型的总结
        """
        # 1. 提取原始文本
        raw_text = self._extract_raw_text(pdf_path)
        if not raw_text:
            return "未能提取到有效文本内容。"

        # 2. 定制化任务 Prompt
        type_specific_prompts = {
            "机制通路图": "重点总结文中的分子相互作用、信号通路级联、受体与配体关系。请明确：谁是上游，谁是下游，在哪里发生。",
            "实验流程图": "重点总结实验的时间轴和操作步骤。请明确：实验分几个阶段，每个阶段的具体操作和检测手段。",
            "技术路线图": "重点总结研究的逻辑架构。请明确：研究的各个模块是如何衔接的，从准备工作到最终验证的逻辑流。",
            "科普插图": "将复杂的医学原理转化为通俗的描述，重点提取病变位置、表现形式和核心致病因素。"
        }
        
        specific_instruction = type_specific_prompts.get(image_type, "请全面总结文中的医学逻辑。")

        # 3. 构造 LLM 输入
        input_text = f"""你是一位专业的医学绘图顾问。请阅读以下文献片段，并为后续绘制【{image_type}】提取核心信息。
要求：
- {specific_instruction}
- 只保留与绘图相关的视觉化实体和逻辑关系。
- 使用简洁的 Markdown 列表形式输出。

文献内容：
{raw_text[:8000]} 
"""
        
        messages = [
            {"role": "system", "content": "你是一个严谨的医学辅助 AI。"},
            {"role": "user", "content": input_text}
        ]
        
        # 4. 执行推理
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=0.7
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def close(self):
        """释放显存"""
        # 显式删除对象并清理缓存
        model_device = self.model.device
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Qwen3-8B 模型已从设备 {model_device} 卸载并清理显存。")


# ==========================================
# 模块测试代码块
# ==========================================
if __name__ == "__main__":
    # --- 1. 测试配置 ---
    # 确保这些路径在你的本地环境下是正确的
    TEST_MODEL_PATH = "../models/Qwen3-8B-Instruct" 
    TEST_PDF_PATH = "../data/test/test_paper.pdf"
    TEST_TYPE = "机制通路图"

    # 检查路径是否存在
    if not os.path.exists(TEST_MODEL_PATH):
        print(f"❌ 错误：在 {TEST_MODEL_PATH} 未找到 Qwen3 模型")
    elif not os.path.exists(TEST_PDF_PATH):
        print(f"❌ 错误：在 {TEST_PDF_PATH} 未找到测试 PDF 文件")
    else:
        print(f"--- 💡 开始单独测试 pdf_parser 模块 ---")
        
        # 2. 初始化解析器
        parser = Qwen3TextParser(model_path=TEST_MODEL_PATH)
        
        try:
            # 3. 执行解析
            print(f"正在分析文件: {TEST_PDF_PATH}，目标类型: {TEST_TYPE}...")
            result = parser.parse(TEST_PDF_PATH, image_type=TEST_TYPE)
            
            # 4. 打印测试结果
            print("\n" + "="*50)
            print(f"【{TEST_TYPE}】解析总结结果：")
            print("-" * 50)
            print(result)
            print("="*50 + "\n")
            
            # 5. 可选：保存到文件查看
            with open("../data/test/paper_summary.md", "w", encoding="utf-8") as f:
               f.write(result)
            print("✅ 测试结果已保存至../data/test/paper_summary.md")

        except Exception as e:
            print(f"❌ 测试过程中发生异常: {e}")
            
        finally:
            # 6. 必须清理显存
            parser.close()