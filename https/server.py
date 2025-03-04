import http.server
import ssl


def run_server():
    server_address = ("0.0.0.0", 4443)  # 设置 IP 和端口
    handler = http.server.SimpleHTTPRequestHandler

    # 创建 HTTPS 服务器
    httpd = http.server.HTTPServer(server_address, handler)

    # 配置 SSL
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print(f"服务器运行在 https://{server_address[0]}:{server_address[1]}/")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("服务器已停止")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
