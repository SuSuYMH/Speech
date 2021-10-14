import wave as wv
import pylab as plt
import numpy as np


def read_voice_signal():
    '''从文件读取声音信号'''
    wave_path = 'data/shan.wav'
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
    # 最后通过取样点数和取样频率计算出每个取样的时间：
    # arange是在给定区间内均匀地返回值
    timelist = np.arange(0, nframes) * (1.0 / framerate)
    return wave_data, timelist


def image_display():
    '''展示图像'''
    wavdata, wavtime = read_voice_signal()
    plt.title("音频双声道显示")
    plt.subplot(211)
    plt.plot(wavtime, wavdata[0], color='red')
    plt.subplot(212)
    plt.plot(wavtime, wavdata[1])
    plt.show()


def pre_emphasis(wavedata):
    '''预加重'''
    # 设置系数
    coefficient = 0.97
    # 通过map预加重
    new_wavedata = np.delete(wavedata[0], 0)
    left_wavedata = np.array(map(lambda x, y: x-coefficient*y, wavedata[0], new_wavedata))
    new_wavedata = np.delete(wavedata[1], 0)
    right_wavedata = np.array(map(lambda x, y: x - coefficient * y, wavedata[1], new_wavedata))
    return left_wavedata, right_wavedata

def main():
    image_display()


if __name__=='__main__':
    main()


