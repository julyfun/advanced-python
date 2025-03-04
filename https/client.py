import requests


def make_request():
    server_url = "https://127.0.0.1:4443"

    try:
        # 警告：在生产环境中应该不使用 verify=False
        # 这里仅用于测试自签名证书
        response = requests.get(server_url, verify=False)
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")


if __name__ == "__main__":
    # 禁止不安全连接警告（仅用于测试）
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    make_request()
