import wave as wv
import pylab as plt
import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft,ifft


def read_voice_signal():
    """从文件读取声音信号"""
    wave_path = 'data/media1.wav'
    file = wv.open(wave_path, 'rb')

    # a = file.getparams().nframes  # 采样总数
    # f = file.getparams().framerate  # 采样频率
    # sample_time = 1 / f  # 采样点的时间间隔
    # timelist = np.arange(0, a) * sample_time # 每一个取样的时间点
    # time = a / f  # 声音信号的总时间

    params = file.getparams()
    # 声道数，量化位数，采样频率（如44100HZ），采样点个数
    nchannels, sampwidth, framerate, nframes = params[:4]

    # 读为byte object型文件
    byteobj_data = file.readframes(nframes)
    file.close()

    # 转化为np的数组
    wave_data = np.frombuffer(byteobj_data, dtype=np.short)
    # 读到的是LRLRLRLRLR的，转为左声道一列，右声道一列
    wave_data.shape = -1, 2
    # 一行
    wave_data = wave_data.T
    # 最后通过取样点数和取样频率计算出每个取样的时间timelist
    # arange是在给定区间内均匀地返回值
    timelist = np.arange(0, nframes) * (1.0 / framerate)
    return wave_data, timelist, framerate, nframes


def frequence_domain_display(fredata, num, f):
    """展示频域图像"""
    timelist = np.arange(0, num) * (f / num)

    normalization_y = np.abs(fredata) / num
    plt.plot(timelist, normalization_y, color='red')
    plt.show()


def time_domain_display(wavdata, num, f):
    """展示时域图像"""
    timelist = np.arange(0, num) * (1.0 / f)
    plt.plot(timelist, wavdata, color='red')
    plt.show()


def image_display(wavdata, wavtime):
    """展示图像"""
    plt.title("音频双声道显示")
    plt.subplot(211)
    plt.plot(wavtime, wavdata[0], color='red')
    plt.subplot(212)
    plt.plot(wavtime, wavdata[1])
    plt.show()


def pre_emphasis(wavedata):
    """预加重"""
    # 设置系数
    coefficient = 0.97
    # 通过map预加重
    new_wavedata = np.delete(wavedata[0], 0)
    left_wavedata = np.array(map(lambda x, y: x - coefficient * y, wavedata[0], new_wavedata))
    new_wavedata = np.delete(wavedata[1], 0)
    right_wavedata = np.array(map(lambda x, y: x - coefficient * y, wavedata[1], new_wavedata))
    return left_wavedata, right_wavedata


def windows(wavedata, timelist, f):
    """分帧与加窗"""
    # 每帧有多少个采样点
    wlen = int(25 * f / 1000)
    # 每次移动frame移动多少个采样点
    inc = int(10 * f / 1000)
    # 重叠部分
    overlap = wlen - inc
    # 算出frame的个数 注意这里一定要用某一个声道的求数量，不然用双声道的，他认为向量个数为2，我赣！
    signal_length = len(wavedata[0])
    if signal_length <= wlen:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = int(np.ceil((1.0 * signal_length - wlen + inc) / inc))
    print(nf)
    # 所有帧加起来总的铺平后的长度
    pad_length = int((nf - 1) * inc + wlen)
    # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    zeros = np.zeros((pad_length - signal_length,))
    # 填补后的信号记为pad_signal
    pad_signal = np.concatenate((wavedata[0], zeros))
    # 相当于对所有帧的时间点进行抽取，得到nf*wlen长度的矩阵
    indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
    # print(indices)
    # nf行，wlen列.每一行就是每一frame的取样点的序号，从零开始
    # 将indices转化为矩阵
    indices = np.array(indices, dtype=np.int32)
    print(indices.shape)
    # 得到帧信号,比如frames[0]就是第一个frame的25个量化值的向量
    frames = pad_signal[indices]
    # 调用汉明窗
    windown = np.hanning(wlen)
    # 创建一个新的numpyarray
    frames_afterwindows = np.zeros((1, int(25 * f/1000)))
    # print(frames_afterwindows)
    # 对每一frame依次加窗
    for i in frames:
        i = i * windown
        # print(i)
        i = np.expand_dims(i, axis=0)  # 升维，才能进行下一步的拼接
        # 把每一步的出来的经过windows的frame作为新的一行添加到frames_afterwindows里
        frames_afterwindows = np.concatenate([frames_afterwindows, i])
        # np.append(frames_afterwindows, i, axis=0)# axis=0 增加行，列数不变
    # 把一开始为了创建一个向量用的玩意去掉
    np.delete(frames_afterwindows, 0)
    # 返回经过windows的frame的矩阵，每一frame的
    print(frames_afterwindows)
    return frames_afterwindows, wlen, nf


def all_frames_fft(frames_afterwindows, f):
    frames_afterfft = np.zeros((1, int(25 * f / 1000)))
    for i in frames_afterwindows:
        i = fft(i)
        # print(i)
        i = np.expand_dims(i, axis=0)  # 升维，才能进行下一步的拼接
        # 把每一步的出来的经过windows的frame作为新的一行添加到frames_afterwindows里
        frames_afterfft = np.concatenate([frames_afterfft, i])
    np.delete(frames_afterfft, 0)
    return frames_afterfft




def main():
    wave_data, timelist, f, sum = read_voice_signal()
    print(f)
    # image_display()
    frames_afterwindows, wlen, nf = windows(wave_data, timelist, f)
    time_domain_display(frames_afterwindows[15], wlen, f)
    frames_afterfft = all_frames_fft(frames_afterwindows, f)
    # print(frames_afterfft[800])
    frequence_domain_display(np.abs(frames_afterfft[130]), wlen, f)


if __name__ == '__main__':
    main()

