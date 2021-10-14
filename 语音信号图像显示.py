import wave as wv
import numpy as np
import matplotlib.pyplot as plt


def wavread(path):
    wavfile = wv.open(path, "rb")
    # 获得参数
    params = wavfile.getparams()
    framesra, frameswav = params[2], params[3]
    # 读取所有采样点得到byte object
    datawav = wavfile.readframes(frameswav)
    wavfile.close()

    # 把读取的二进制音频转化为矩阵形式 得到numpy.ndarray
    datause = np.frombuffer(datawav, dtype=np.short)
    # print(datause.type)
    datause.shape = -1, 2
    datause = datause.T
    time = np.arange(0, frameswav) * (1.0 / framesra)
    return datause, time


def main():
    # path = input("The Path is:")
    path = '/data/shan.wav'
    wavdata, wavtime = wavread(path)
    plt.title("shan.wav's Frames")
    plt.subplot(211)
    plt.plot(wavtime, wavdata[0], color='red')
    plt.subplot(212)
    plt.plot(wavtime, wavdata[1])
    plt.show()


if __name__=='__main__':
    main()