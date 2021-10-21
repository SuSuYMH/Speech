import wave as wv
import pylab as plt
import numpy as np
from scipy.fftpack import fft, idct


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


def image_display(data, timelist, title):
    """展示任意图像"""
    plt.title(title)
    plt.plot(timelist, data, color='red')
    plt.show()


# def time_domain_display(wavdata, num, f):
#     """展示时域图像"""
#     timelist = np.arange(0, num) * (1.0 / f)
#     plt.plot(timelist, wavdata, color='red')
#     plt.show()


# def image_display(wavdata, wavtime):
#     """展示图像"""
#     plt.title("音频双声道显示")
#     plt.subplot(211)
#     plt.plot(wavtime, wavdata[0], color='red')
#     plt.subplot(212)
#     plt.plot(wavtime, wavdata[1])
#     plt.show()


def pre_emphasis(wavedata, sum):
    """预加重"""
    # 设置系数
    coefficient = 0.97
    # 预加重
    new_wavedata = np.delete(wavedata, 0)
    new_wavedata = np.append(new_wavedata, [0], axis=0)
    singleside_wavedata = wavedata - new_wavedata * coefficient
    return singleside_wavedata


def windows(L_wave_data, timelist, f):
    """分帧与加窗"""
    # 每帧有多少个采样点
    wlen = int(25 * f / 1000)
    # 每次移动frame移动多少个采样点
    inc = int(10 * f / 1000)
    # 重叠部分
    overlap = wlen - inc
    # 算出frame的个数 注意这里一定要用某一个声道的求数量，不然用双声道的，他认为向量个数为2，我赣！
    signal_length = len(L_wave_data)
    if signal_length <= wlen:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = int(np.ceil((1.0 * signal_length - wlen + inc) / inc))
    # print(nf)
    # 所有帧加起来总的铺平后的长度
    pad_length = int((nf - 1) * inc + wlen)
    # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    zeros = np.zeros((pad_length - signal_length,))
    # 填补后的信号记为pad_signal
    pad_signal = np.concatenate((L_wave_data, zeros))
    # 相当于对所有帧的时间点进行抽取，得到nf*wlen长度的矩阵
    indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
    # print(indices)
    # nf行，wlen列.每一行就是每一frame的取样点的序号，从零开始
    # 将indices转化为矩阵
    indices = np.array(indices, dtype=np.int32)
    # print('（frame数，一frame点数）')
    # print(indices.shape)
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
    '''一定要在赋值回来！！！，而且记得要axis=0！！！不然会变成向量！！！'''
    frames_afterwindows = np.delete(frames_afterwindows, 0, axis = 0)

    # 返回经过windows的frame的矩阵，每一frame的
    # print(frames_afterwindows)
    return frames_afterwindows, wlen, nf


def all_frames_fft(frames_afterwindows, num, f):
    # windows后的图片，frame个数，频率
    frames_afterfft = np.zeros((1, int(25 * f / 1000)))
    for i in frames_afterwindows:
        i = fft(i)
        # print(i)
        i = np.expand_dims(i, axis=0)  # 升维，才能进行下一步的拼接
        # 把每一步的出来的经过windows的frame作为新的一行添加到frames_afterwindows里
        frames_afterfft = np.concatenate([frames_afterfft, i])
    frames_afterfft = np.delete(frames_afterfft, 0, axis = 0)
    # 按找基频的倍数，一直延伸到可接受的最高频率即采样频率，但是由于三角函数的一些关系，会使fft后的频域图像对称分布，而
    # 高频那部分的数据是和低频对称的，高频那部分是假的！
    '''能量谱的取一半是把每一帧的取一半，而不是把帧数取一半'''
    timelist = np.arange(0, num) * (f / num)    # 从零频率到最高采样频率
    half_timelist = timelist[range(int(num / 2))]  # 取一半区间
    # 对原频率取模为能量谱
    normalization_frames_afterabs = (np.abs(frames_afterfft))**2/ num
    # print(len(normalization_frames_afterabs))
    normalization_half_frames_afterabs = normalization_frames_afterabs[:, range(int(num / 2))]  # 由于对称性，只取一半区间（单边频谱）。根据Nyquist定理，超过采样率一半的信号会出现混叠
    # print(len(normalization_half_frames_afterabs))
    # 取每一帧的能量和的10倍log(最后mfcc特征，添加到十二维的最后一维变为十三维)
    frames_afterfft_linshi = np.where(normalization_half_frames_afterabs == 0, np.finfo(float).eps, normalization_half_frames_afterabs)
    energy_13 = 10*np.log10(frames_afterfft_linshi.sum(axis=1))
    # print('energy_13!!!!')
    # print(energy_13.shape)
    # 返回值各个是什么看上面注释，太不好描述了
    return frames_afterfft, timelist, normalization_half_frames_afterabs, half_timelist, energy_13


def melfilter(half_frames_afterabs, nf, f):
    # fft后的数据，帧数，频率
    # 滤波器个数
    nfilt = 26
    low_freq_mel = 0
    high_freq_mel = (1127 * np.log(1 + (f / 2) / 700))  # 把 Hz 变成 Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 将梅尔刻度等间隔
    hz_points = (700 * (np.e ** (mel_points / 1127) - 1))  # 把 Mel 变成 Hz
    # 每一个滤波器的中间值bin
    bin = np.floor((nf + 1) * hz_points / f)
    # 每一个半能量谱的滤波器
    fbank = np.zeros((nfilt, int(np.floor(nf / 2))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    # print(fbank[0])
    # 获得的是frame数*mel滤波器个数的矩阵，每一行就是该帧的mel-scale power spectrum
    melscale_power_spectrum = np.dot(half_frames_afterabs[0:nf], fbank.T)
    # print(filter_banks.shape)
    # print(filter_banks)
    # 不能计算log(0),所以处理一下
    melscale_power_spectrum = np.where(melscale_power_spectrum == 0, np.finfo(float).eps, melscale_power_spectrum)  # 数值稳定性
    # print(filter_banks)
    # 进行log处理
    melscale_power_spectrum = 10 * np.log10(melscale_power_spectrum)  # dB
    # print(filter_banks)
    timely_T = melscale_power_spectrum.T
    timely_T -= (np.mean(timely_T, axis=0) + 1e-8)
    melscale_power_spectrum = timely_T.T
    # print(filter_banks)
    return melscale_power_spectrum, nfilt


def all_melframes_idct(melscale_power_spectrum, nf, nfilt):
    melframes_afteridct = np.zeros((1, nfilt))
    for i in melscale_power_spectrum:
        i = idct(i)
        # print(i)
        i = np.expand_dims(i, axis=0)  # 升维，才能进行下一步的拼接
        # 把每一步的出来的经过windows的frame作为新的一行添加到frames_afterwindows里
        melframes_afteridct = np.concatenate([melframes_afteridct, i])
    melframes_afteridct = np.delete(melframes_afteridct, 0, axis = 0)
    # 每一个滤波器得到的值再经过离散余弦变换之后对应一个系数，取前面n个，n一般取12
    feature = 12
    # 返回每一帧的feature和feature的维数
    return melframes_afteridct[:, range(feature)], feature


def order_difference(melframes_afteridct, dim_of_feature, times_of_order, energy_13):
    # 输入为系数、维数、要做几阶差分、还有能量
    # print(melframes_afteridct[1])
    # print('energy_13!!!!')
    # print(energy_13.shape)
    energy_13 = np.expand_dims(energy_13, axis=0)
    # print('energy_13!!!!')
    # print(energy_13.shape)
    # energy_13 = np.expand_dims(energy_13, axis=1)
    melframes_afteridct = np.append(melframes_afteridct, energy_13.T, axis=1)
    # print(melframes_afteridct[1])
    zeros = np.zeros((dim_of_feature + 1,))
    zeros = np.expand_dims(zeros, axis=0)
    # 最终mfcc特征
    final_mfcc_feature = melframes_afteridct
    # 当前得到的差分，先初始化为开始输入的系数
    current_order = melframes_afteridct
    for i in range(times_of_order):
        # print(current_order.shape)
        # print('hhhhhh')
        current_order_unchanged = current_order
        current_order = np.delete(current_order, [0, 1], axis=0)
        # print(current_order.shape)
        current_order = np.append(current_order, zeros, axis=0)
        # print(current_order.shape)
        current_order = np.append(current_order, zeros, axis=0)
        # print(current_order.shape)
        current_order = current_order_unchanged - current_order
        final_mfcc_feature = np.append(final_mfcc_feature, current_order, axis=1)
    return final_mfcc_feature.T, dim_of_feature*(times_of_order+1)


def main():
    """读取数据"""
    num_of_testframe = 14
    # 双声道数据，有数据的时间点，取样频率，取样个数
    wave_data, timelist, f, sum = read_voice_signal()
    # print(f)
    # image_display()

    '''预加重'''
    # 之后分左右声道分别处理
    L_wave_data = pre_emphasis(wave_data[0], sum)
    R_wave_data = pre_emphasis(wave_data[1], sum)
    # print(L_wave_data.shape)
    # print('预加重后形状')

    '''加窗'''
    # 加窗后的数据，每帧的采样点个数，帧数
    frames_afterwindows, wlen, nf = windows(L_wave_data, timelist, f)
    # 对于每一frame来说的有数据的时间点
    timelist_time_frame = np.arange(0, wlen) * (1.0 / f)
    image_display(frames_afterwindows[num_of_testframe], timelist_time_frame, 'after_windows')
    # print(frames_afterwindows.shape)
    # print('加窗后形状')

    '''fft'''
    # fft后的数据,及其平方并取前半个图的能量数据
    frames_afterfft, timelist, normalization_half_frames_afterabs, timelist_fre_frame, energy_13 = all_frames_fft(frames_afterwindows, wlen, f)
    # print(frames_afterfft[800])
    image_display(normalization_half_frames_afterabs[num_of_testframe], timelist_fre_frame, "energy_afterfft")
    # print(normalization_half_frames_afterabs.shape)
    # print('fft且abs后形状')

    '''mel滤波'''
    melscale_power_spectrum, nfilt = melfilter(normalization_half_frames_afterabs, wlen, f)
    # 高频的那部分降下去很正常，因为高频就没什么能量
    image_display(melscale_power_spectrum[num_of_testframe], np.arange(0, nfilt), "after_mel")
    # print(melscale_power_spectrum.shape)
    # print('mel滤过后')

    '''idct'''
    melframes_afteridct, dim_of_feature = all_melframes_idct(melscale_power_spectrum, nf, nfilt)
    # print(melframes_afteridct.T.shape)
    image_display(melframes_afteridct[num_of_testframe], np.arange(0, dim_of_feature), "after_idct")
    # print('离散余弦变换后')

    '''difference差分'''
    # 要进行几阶差分
    times_of_order = 2
    # 最后返回按列排的特征，及其维数
    final_mfcc_feature, dim_of_final_feature = order_difference(melframes_afteridct, dim_of_feature, times_of_order, energy_13)
    print(final_mfcc_feature[:, 13])
    # print(final_mfcc_feature.shape)
    image_display(final_mfcc_feature[:, 13], np.arange(0, 39), "mfcc")


if __name__ == '__main__':
    main()

