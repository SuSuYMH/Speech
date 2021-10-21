import os
import wave as wv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from matplotlib.pylab import mpl


def main():
    # fs = 44100
    # high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # 把 Hz 变成 Mel
    # mel_points = np.linspace(0, high_freq_mel, 26 + 2)  # 将梅尔刻度等间隔
    # hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # 把 Mel 变成 Hz
    # bin = np.floor((1024 + 1) * hz_points / fs)
    # print(bin)
    # fbank = np.zeros((24, int(np.floor(149 / 2 + 1))))
    # print(fbank.shape)

    wavedata = np.array([[1, 2, 3, 4],[2, 3, 4, 5]])
    print(np.mean(wavedata,axis=0))
    # print(wavedata.sum(axis = 1))
    # num = wavedata[0].sum
    # print(num)
    # print(range(int(10 / 2)))

    # new_wavedata = np.delete(wavedata, 0)
    # print(new_wavedata)
    # new_wavedata = np.append(wavedata, [0], axis=0)
    # print(new_wavedata)

    # n = np.array([complex(1, 2), complex(2, 3)])
    # m = np.abs(n)
    # print(n)
    # print(m)

    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    # mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
    #
    # # 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
    # x = np.linspace(0, 1, 1400)
    #
    # # 设置需要采样的信号，频率分量有200，400和600
    # y = 7 * np.sin(2 * np.pi * 200 * x) + 5 * np.sin(2 * np.pi * 400 * x) + 3 * np.sin(2 * np.pi * 600 * x)
    #
    # fft_y = fft(y)  # 快速傅里叶变换
    #
    # N = 1400
    # x = np.arange(N)  # 频率个数
    # half_x = x[range(int(N / 2))]  # 取一半区间
    #
    # abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    # angle_y = np.angle(fft_y)  # 取复数的角度
    # normalization_y = abs_y / N  # 归一化处理（双边频谱）
    # normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
    #
    # plt.subplot(231)
    # plt.plot(x, y)
    # plt.title('原始波形')
    #
    # plt.subplot(232)
    # plt.plot(x, fft_y, 'black')
    # plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')
    #
    # plt.subplot(233)
    # plt.plot(x, abs_y, 'r')
    # plt.title('双边振幅谱(未归一化)', fontsize=9, color='red')
    #
    # plt.subplot(234)
    # plt.plot(x, angle_y, 'violet')
    # plt.title('双边相位谱(未归一化)', fontsize=9, color='violet')
    #
    # plt.subplot(235)
    # plt.plot(x, normalization_y, 'g')
    # plt.title('双边振幅谱(归一化)', fontsize=9, color='green')
    #
    # plt.subplot(236)
    # plt.plot(half_x, normalization_half_y, 'blue')
    # plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
    #
    # plt.show()

    # print(np.array([1,2,3])*np.array([2,3,4]))
    # # 帧长
    # wlen = 25
    # # 帧移
    # inc = 10
    # nf = 10
    # indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
    # print(indices)
    # # 将indices转化为矩阵
    # indices = np.array(indices, dtype=np.int32)
    # print(indices)
    # # 得到帧信号
    # # frames = pad_signal[indices]
    # # 调用汉明窗
    # windown = np.hanning(wlen)

    wave_path = 'data/shan.wav'
    # s = '12345678'
    # array = np.fromstring(s, dtype=np.short)
    # print(array)

    # 以二进制文件打开
    # file = open("shan.wav", "rb")
    # s = file.read(44)
    # print(s)

    # 尝试getparams函数
    # shanWav = wv.open(wave_path, 'rb')
    # params = shanWav.getparams()
    # print(params[2])
    # windows(shanWav)

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
# def windows(wavedata):
#     """分帧与加窗"""
#     # 帧长
#     wlen = 25
#     # 帧移
#     inc = 10
#     # 重叠部分
#     overlap = wlen - inc
#     # 算出frame的个数
#     signal_length = len(wavedata)
#     if signal_length <= wlen:  # 若信号长度小于一个帧的长度，则帧数定义为1
#         nf = 1
#     else:  # 否则，计算帧的总长度
#         nf = int(np.ceil((1.0 * signal_length - wlen + inc) / inc))
#     # 所有帧加起来总的铺平后的长度
#     pad_length = int((nf - 1) * inc + wlen)
#     # 不够的长度使用0填补，类似于FFT中的扩充数组操作
#     zeros = np.zeros((pad_length - signal_length,))
#     # 填补后的信号记为pad_signal
#     pad_signal = np.concatenate((wavedata, zeros))
#     # 相当于对所有帧的时间点进行抽取，得到nf*wlen长度的矩阵
#     indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
#     print(indices)

if __name__ == '__main__':
    main()
