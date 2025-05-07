import pyaudio
import wave
import time


def record_audio(
    filename, seconds=5, sample_rate=44100, channels=1, format_type=pyaudio.paInt16
):
    """
    录制麦克风音频

    参数:
        filename (str): 保存的文件名
        seconds (int): 录制时长(秒)
        sample_rate (int): 采样率
        channels (int): 通道数
        format_type: 音频格式
    """
    # 初始化PyAudio
    p = pyaudio.PyAudio()

    # 设置音频参数
    chunk = 1024  # 每个缓冲区的帧数

    print(f"开始录制，将录制 {seconds} 秒...")

    # 打开音频流
    stream = p.open(
        format=format_type,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk,
    )

    # 存储录制的帧
    frames = []

    # 录制音频
    for i in range(0, int(sample_rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
        # 显示录制进度
        if i % 10 == 0:
            progress = i / (int(sample_rate / chunk * seconds)) * 100
            print(f"\r录制进度: {progress:.1f}%", end="")

    print("\n录制完成!")

    # 停止和关闭音频流
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 保存为WAV文件
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format_type))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()

    print(f"音频已保存为: {filename}")


if __name__ == "__main__":
    # 录制5秒音频并保存
    record_audio("recorded_audio.wav", seconds=5)
