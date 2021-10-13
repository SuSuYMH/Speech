'''
功能：读取指定文件夹下的所有mp3文件，转换为wav文件
用法：修改path路径 filter改为".m4a",就是遍历当前目录下的m4a文件，注释掉35行的cmd_command
转换其它的：最主要是用字符串拼接出cmd_command命令

'''

import os
import pydub
import ffmpeg

path=r'/Users/su/Desktop/数据/语音数据'

def use_ffmpeg():
    filter=[".m4a"] #设置过滤后的文件类型 当然可以设置多个类型
    filenames = all_path(path)

    for filename in filenames:
        filename = str(filename)
        temp = filename.split('.')

        # 将.m4a格式转为wav格式的命令
        cmd_command = "ffmpeg -i {0} -acodec pcm_s16le -ac 1 -ar 16000 -y {1}.wav && del {0}".format(filename, temp[0])
        # 将.mp3格式转为wav格式的命令
        # cmd_command = "ffmpeg -loglevel quiet -y -i {0} -ar 16000 -ac 1 {1}.wav && del {0}".format(filename, temp[0])

        # print(cmd_command)
        os.system(cmd_command)


def all_path(dirname):

    result = []#所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):

        for filename in file_name_list:
            apath = os.path.join(maindir, filename)#合并成一个完整路径
            ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容

            if ext in filter:
                result.append(apath)

    return result

#def hehe():
#    sound = AudioSegment.from_m4a(path)
#   sound.export(path, format='wav')

if __name__=='__main__':
    use_ffmpeg()