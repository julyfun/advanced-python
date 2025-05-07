from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uvicorn
from typing import List
import shutil

# 创建 FastAPI 应用
app = FastAPI(title="AR数据采集服务器", description="HTTP版本的AR数据采集服务器")

# 确保上传目录存在
uploads_dir = "uploads"
os.makedirs(uploads_dir, exist_ok=True)

# 提供静态文件服务
# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """提供首页，显示服务器状态和上传的文件列表"""

    # 构建 HTML 页面
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AR数据采集服务器</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .status { padding: 10px; background-color: #e6f7ff; border-radius: 5px; }
            .files { margin-top: 20px; }
            .file-item { padding: 8px; border-bottom: 1px solid #eee; }
        </style>
    </head>
    <body>
        <h1>AR数据采集服务器 (HTTP版)</h1>
        <div class="status">服务器状态: 运行中</div>
        
        <div class="files">
            <h2>已上传文件:</h2>
    """

    # 列出uploads目录中的文件
    if os.path.exists(uploads_dir):
        files = os.listdir(uploads_dir)
        if files:
            for file in files:
                file_path = os.path.join(uploads_dir, file)
                file_size = os.path.getsize(file_path)
                html_content += (
                    f'<div class="file-item">{file} ({format_size(file_size)})</div>\n'
                )
        else:
            html_content += '<div class="file-item">暂无上传文件</div>\n'
    else:
        html_content += '<div class="file-item">上传目录不存在</div>\n'

    html_content += """
        </div>
    </body>
    </html>
    """

    return html_content


@app.post("/upload", response_class=JSONResponse)
async def upload_file(file: UploadFile = File(...)):
    """处理文件上传"""
    try:
        # 保存上传的文件
        file_path = os.path.join(uploads_dir, file.filename)

        # 使用 shutil 将上传的文件内容写入磁盘
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 返回成功响应
        return {
            "status": "success",
            "message": f"文件 {file.filename} 上传成功",
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"上传失败: {str(e)}"},
        )


def format_size(size):
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


if __name__ == "__main__":
    # 启动服务器，监听所有接口，使用8000端口
    print(f"服务器运行在 http://0.0.0.0:8000/")
    print("支持文件上传: http://0.0.0.0:8000/upload")
    print("访问首页查看上传文件列表: http://0.0.0.0:8000/")
    print("按 Ctrl+C 停止服务器")

    uvicorn.run(app, host="0.0.0.0", port=8000)
