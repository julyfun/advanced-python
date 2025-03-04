import http.server
import ssl
import os
import cgi
import json
from urllib.parse import parse_qs


class UploadHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # 处理GET请求，提供简单的状态页面
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()

            # 简单的HTML页面，显示服务器状态和上传的文件列表
            response = """
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
                <h1>AR数据采集服务器</h1>
                <div class="status">服务器状态: 运行中</div>
                
                <div class="files">
                    <h2>已上传文件:</h2>
            """

            # 列出uploads目录中的文件
            uploads_dir = "uploads"
            if os.path.exists(uploads_dir):
                files = os.listdir(uploads_dir)
                if files:
                    for file in files:
                        file_path = os.path.join(uploads_dir, file)
                        file_size = os.path.getsize(file_path)
                        response += f'<div class="file-item">{file} ({self.format_size(file_size)})</div>\n'
                else:
                    response += '<div class="file-item">暂无上传文件</div>\n'
            else:
                response += '<div class="file-item">上传目录不存在</div>\n'

            response += """
                </div>
            </body>
            </html>
            """

            self.wfile.write(response.encode("utf-8"))
        else:
            # 对于其他路径，使用默认的处理方法
            super().do_GET()

    def do_POST(self):
        # 处理POST请求，用于文件上传
        if self.path == "/upload":
            # 确保上传目录存在
            uploads_dir = "uploads"
            os.makedirs(uploads_dir, exist_ok=True)

            # 解析表单数据
            content_type, pdict = cgi.parse_header(self.headers["Content-Type"])

            if content_type == "multipart/form-data":
                # 处理multipart/form-data请求
                pdict["boundary"] = pdict["boundary"].encode("utf-8")
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers["Content-Type"],
                    },
                )

                # 检查是否有文件字段
                if "file" in form:
                    fileitem = form["file"]

                    # 检查是否是文件字段
                    if fileitem.filename:
                        # 保存文件
                        filepath = os.path.join(
                            uploads_dir, os.path.basename(fileitem.filename)
                        )
                        with open(filepath, "wb") as f:
                            f.write(fileitem.file.read())

                        # 返回成功响应
                        self.send_response(200)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()

                        response = {
                            "status": "success",
                            "message": f"文件 {fileitem.filename} 上传成功",
                            "file_path": filepath,
                            "file_size": os.path.getsize(filepath),
                        }

                        self.wfile.write(
                            json.dumps(response, ensure_ascii=False).encode("utf-8")
                        )
                        return

            # 如果没有文件或处理失败
            self.send_response(400)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            response = {"status": "error", "message": "上传失败，未找到有效的文件"}

            self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
        else:
            # 对于其他路径，返回404
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            response = {"status": "error", "message": "请求的路径不存在"}

            self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))

    def format_size(self, size):
        # 格式化文件大小
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"


def run_server():
    server_address = ("0.0.0.0", 4443)  # 设置 IP 和端口

    # 使用自定义的处理器
    httpd = http.server.HTTPServer(server_address, UploadHandler)

    # 配置 SSL
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print(f"服务器运行在 https://{server_address[0]}:{server_address[1]}/")
    print("支持文件上传: https://{0}:{1}/upload".format(*server_address))
    print("按 Ctrl+C 停止服务器")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
