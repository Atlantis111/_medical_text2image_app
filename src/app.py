import os
import time
import gradio as gr
from config import ASPECT_RATIO_MAP
from pdf_parser import Qwen3TextParser
from llm_processor import LocalPromptGenerator
from image_generator import MedicalImageGenerator

def process_pipeline(pdf_file, user_query, image_type, style, language, density, aspect_ratio):
    """
    串行执行管线：解析 PDF -> 生成 Prompt -> 生成图片
    每个步骤完成后立刻释放显存，保证系统稳定。
    """
    if not pdf_file:
        yield "⚠️ 请先上传 PDF 文件", "未开始", None
        return

    pdf_path = pdf_file.name
    
    # 确保输出目录存在
    output_dir = os.path.abspath("../data/output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())

    try:
        # ==========================================
        # 步骤 1: 解析 PDF 内容
        # ==========================================
        yield "⏳ [1/3] 正在加载 Qwen3-8B 解析 PDF 内容，请稍候...", "等待中...", None
        parser = Qwen3TextParser()
        pdf_summary = parser.parse(pdf_path, image_type)
        parser.close() # 严格清理显存
        
        # ==========================================
        # 步骤 2: 生成生图提示词 (Prompt)
        # ==========================================
        yield f"✅ PDF 解析完成！提取内容如下：\n\n{pdf_summary}", "⏳ [2/3] 正在加载模型生成专属绘画提示词...", None
        prompt_gen = LocalPromptGenerator()
        image_prompt = prompt_gen.generate(user_query, pdf_summary, style, language, density, image_type)
        prompt_gen.close() # 严格清理显存

        # ==========================================
        # 步骤 3: 绘制并生成图片
        # ==========================================
        yield (
            f"✅ PDF 解析完成！提取内容如下：\n\n{pdf_summary}", 
            f"✅ 提示词生成完成：\n\n{image_prompt}\n\n⏳ [3/3] 正在加载 Qwen-Image 引擎生成图像，此过程可能需要几分钟...", 
            None
        )
        w, h = ASPECT_RATIO_MAP.get(aspect_ratio, (1024, 1024))
        output_img_path = os.path.join(output_dir, f"medical_img_{timestamp}.jpg")
        
        img_gen = MedicalImageGenerator()
        final_img_path = img_gen.generate(image_prompt, image_type, width=w, height=h, output_path=output_img_path)
        img_gen.close() # 严格清理显存

        # ==========================================
        # 完成：返回最终结果
        # ==========================================
        yield (
            f"✅ PDF 解析完成！提取内容如下：\n\n{pdf_summary}", 
            f"✅ 提示词生成完成：\n\n{image_prompt}", 
            final_img_path
        )

    except Exception as e:
        yield f"❌ 发生错误：{str(e)}", "运行中断", None

# --- 构建 Gradio 页面 ---
with gr.Blocks(title="AI 医学文献绘图系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧬 AI 医学文献绘图系统")
    gr.Markdown("上传医学文献 PDF，输入您的具体要求，系统将自动解析文献逻辑并生成专业医学配图。")

    with gr.Row():
        # 左侧：输入控制面板
        with gr.Column(scale=1):
            file_input = gr.File(label="📄 上传文献 (PDF格式)", file_types=[".pdf"])
            query_input = gr.Textbox(label="✍️ 补充要求 (自然语言)", placeholder="例如：我想突出展示姜黄素对线粒体的影响，整体色调偏蓝色...", lines=3)
            
            with gr.Row():
                type_input = gr.Dropdown(choices=["机制通路图", "实验流程图", "技术路线图", "科普插图"], value="机制通路图", label="🖼️ 图片类型")
                ratio_input = gr.Dropdown(choices=["1:1", "16:9", "4:3", "3:4"], value="1:1", label="📐 画面比例")
            
            with gr.Row():
                style_input = gr.Dropdown(choices=["3D render, medical-grade, macro photography", "Flat vector illustration, clean lines", "Watercolor scientific style", "Neon glowing futuristic style"], value="3D render, medical-grade, macro photography", label="🎨 艺术风格", allow_custom_value=True)
                lang_input = gr.Dropdown(choices=["English Labels", "中文标签"], value="English Labels", label="🌐 文字/标签语言")
            
            density_input = gr.Radio(choices=["Low", "Medium", "High"], value="Medium", label="🧩 画面元素密度")
            
            submit_btn = gr.Button("🚀 一键开始生成", variant="primary")

        # 右侧：结果展示面板
        with gr.Column(scale=1):
            img_output = gr.Image(label="✨ 生成结果", type="filepath")
            
            with gr.Accordion("🔍 查看中间解析过程", open=False):
                summary_output = gr.Textbox(label="1. 文献核心总结", lines=5, interactive=False)
                prompt_output = gr.Textbox(label="2. AI 生成的底层 Prompt", lines=5, interactive=False)

    # 绑定事件
    submit_btn.click(
        fn=process_pipeline,
        inputs=[file_input, query_input, type_input, style_input, lang_input, density_input, ratio_input],
        outputs=[summary_output, prompt_output, img_output]
    )

if __name__ == "__main__":
    # 启动应用，允许局域网内访问
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)