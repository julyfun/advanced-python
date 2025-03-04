import requests
import os
import argparse
import json


def upload_file(file_path, server_url="https://47.103.61.134:4443"):
    """
    向HTTPS服务器上传文件

    Args:
        file_path: 要上传的文件路径
        server_url: 服务器URL地址

    Returns:
        服务器响应
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return None

    # 构建上传URL
    upload_url = f"{server_url}/upload"

    try:
        # 准备文件
        file_name = os.path.basename(file_path)
        files = {"file": (file_name, open(file_path, "rb"))}

        print(f"正在上传文件: {file_name}")
        print(f"文件大小: {format_size(os.path.getsize(file_path))}")
        print(f"上传到: {upload_url}")

        # 发送POST请求上传文件，忽略SSL证书验证
        response = requests.post(upload_url, files=files, verify=False)

        # 关闭文件
        files["file"][1].close()

        if response.status_code == 200:
            # 解析JSON响应
            try:
                result = response.json()
                print("\n上传成功!")
                print(f"服务器消息: {result.get('message')}")
                if "file_size" in result:
                    print(f"文件大小: {format_size(result['file_size'])}")
                return result
            except json.JSONDecodeError:
                print(f"服务器返回了非JSON格式的响应: {response.text}")
        else:
            print(f"上传失败，HTTP状态码: {response.status_code}")
            print(f"错误信息: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

    return None


def format_size(size):
    """格式化文件大小显示"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="向HTTPS服务器上传文件")
    parser.add_argument("file", type=str, help="要上传的文件路径")
    parser.add_argument(
        "--server",
        type=str,
        default="https://47.103.61.134:4443",
        help="服务器URL地址 (默认: https://47.103.61.134:4443)",
    )
    args = parser.parse_args()

    # 禁止不安全连接警告
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # 上传文件
    upload_file(args.file, args.server)


if __name__ == "__main__":
    main()
