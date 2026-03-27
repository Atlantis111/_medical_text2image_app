# main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from worker import celery_app, process_medical_pipeline

app = FastAPI(title="AI 医学文献绘图 API", version="1.0")

UPLOAD_DIR = os.path.abspath("../data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/v1/task/create")
async def create_task(
    file: UploadFile = File(...),
    user_query: str = Form(""),
    image_type: str = Form("机制通路图"),
    style: str = Form("3D render, medical-grade, macro photography"),
    language: str = Form("English Labels"),
    density: str = Form("Medium"),
    aspect_ratio: str = Form("1:1")
):
    """
    接收 APP 传来的文件和参数，创建异步绘画任务
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只允许上传 PDF 文件")

    # 1. 保存上传的文件到本地（实际生产中应上传至 OSS）
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. 将任务推送到 Celery 队列
    task = process_medical_pipeline.delay(
        pdf_path=file_path,
        user_query=user_query,
        image_type=image_type,
        style=style,
        language=language,
        density=density,
        aspect_ratio=aspect_ratio
    )

    # 3. 立即返回 Task ID 给 APP
    return {"message": "任务已提交", "task_id": task.id}

@app.get("/api/v1/task/status/{task_id}")
async def get_task_status(task_id: str):
    """
    APP 轮询此接口获取任务进度或最终结果
    """
    task_result = celery_app.AsyncResult(task_id)
    
    if task_result.state == 'PENDING':
        return JSONResponse({"state": task_result.state, "status": "任务正在排队中..."})
        
    elif task_result.state == 'PROGRESS':
        # 返回我们在 worker 中 update_state 写入的进度信息
        return JSONResponse({
            "state": task_result.state,
            "step": task_result.info.get('step', 0),
            "status": task_result.info.get('status', '')
        })
        
    elif task_result.state == 'SUCCESS':
        return JSONResponse({
            "state": task_result.state,
            "result": task_result.result # 包含图片路径和提取的 prompt
        })
        
    else:
        # 处理失败状态 (FAILURE)
        return JSONResponse({
            "state": task_result.state,
            "error": str(task_result.info)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)