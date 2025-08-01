from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/check_stream', methods=['POST'])
def check_stream():
    stream_name = request.form.get('name')

    # 判断是否有新推流
    if is_new_stream(stream_name):
        # 停止旧流
        stop_old_stream(stream_name)
        return jsonify({"code": 0, "message": "Old stream stopped, allowing new publish"})

    return jsonify({"code": 1, "message": "Continue with old stream"})

def is_new_stream(stream_name):
    # 实现逻辑判断是否有新推流
    # 例如检查推流来源或时间戳
    return True  # 假设有新推流

def stop_old_stream(stream_name):
    # 实现停止旧流的逻辑
    # 可以通过发送信号或调用 API 来实现
    print(f"Stopping old stream for {stream_name}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=28887)
