import os
import wave as wv
import numpy as np
import matplotlib.pyplot as plt


def main():
    wave_path = 'data/shan.wav'
    s = '12345678'
    array = np.fromstring(s, dtype=np.short)
    print(array)

    # 以二进制文件打开
    # file = open("shan.wav", "rb")
    # s = file.read(44)
    # print(s)

    # 尝试getparams函数
    # shanWav = wv.open('shan.wav', 'rb')
    # params = shanWav.getparams()
    # print(params)

'''
    file = wv.open(wave_path)
    # print('---------声音信息------------')
    # for item in enumerate(WAVE.getparams()):
    #     print(item)
    a = file.getparams().nframes  # 采样总数
    f = file.getparams().framerate  # 采样频率
    sample_time = 1 / f  # 采样点的时间间隔
    time = a / f  # 声音信号的长度
    # sample_frequency, audio_sequence = wavfile.read(wave_path)
    # print(audio_sequence)  # 声音信号每一帧的“大小”
    x_seq = np.arange(0, time, sample_time)

    plt.plot(x_seq, time, 'blue')
    plt.xlabel("time (s)")
    plt.show()
'''


if __name__ == '__main__':
    main()
