# worker.py
import os
import time
from celery import Celery
from config import ASPECT_RATIO_MAP

# 导入你写好的 AI 核心模块
from pdf_parser import Qwen3TextParser
from llm_processor import LocalPromptGenerator
from image_generator import MedicalImageGenerator

# 配置 Celery，使用 Redis 作为消息代理和状态后端
celery_app = Celery(
    "medical_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

# 强制限制 Celery 工作进程，防止并发导致 OOM 爆显存
celery_app.conf.update(
    worker_concurrency=1,
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json']
)

@celery_app.task(bind=True, name="process_medical_pipeline")
def process_medical_pipeline(self, pdf_path, user_query, image_type, style, language, density, aspect_ratio):
    """
    后台排队执行的核心 Pipeline
    """
    output_dir = os.path.abspath("../data/output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    
    try:
        # --- 步骤 1: 解析 PDF ---
        self.update_state(state='PROGRESS', meta={'step': 1, 'status': '正在解析 PDF 文献...'})
        parser = Qwen3TextParser()
        pdf_summary = parser.parse(pdf_path, image_type)
        parser.close()

        # --- 步骤 2: 生成 Prompt ---
        self.update_state(state='PROGRESS', meta={'step': 2, 'status': '正在生成生图提示词...'})
        prompt_gen = LocalPromptGenerator()
        image_prompt = prompt_gen.generate(user_query, pdf_summary, style, language, density, image_type)
        prompt_gen.close()

        # --- 步骤 3: 生成图像 ---
        self.update_state(state='PROGRESS', meta={'step': 3, 'status': '正在使用 Qwen-Image 绘制图像...'})
        w, h = ASPECT_RATIO_MAP.get(aspect_ratio, (1024, 1024))
        output_img_path = os.path.join(output_dir, f"medical_img_{timestamp}.jpg")
        
        img_gen = MedicalImageGenerator()
        final_img_path = img_gen.generate(image_prompt, image_type, width=w, height=h, output_path=output_img_path)
        img_gen.close()

        return {
            "status": "SUCCESS",
            "summary": pdf_summary,
            "prompt": image_prompt,
            "image_path": final_img_path  # 实际生产中应替换为 OSS 的公网 URL
        }

    except Exception as e:
        # 异常捕获与状态更新
        return {"status": "FAILED", "error": str(e)}