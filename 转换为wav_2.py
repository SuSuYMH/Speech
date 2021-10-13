import os
from threading import Thread


# 在命令行窗口调用命令
def execute_cmd_command(command_word: str):
    os.system(command_word)


'''
###########################################################
注:因为是命令行窗口执行命令,因此以下路径要用绝对路径

变量解释:
    source_audio:  需要更换格式的音频文件路径,这里测试选择m4a
    change_audio:  更换格式后的音频文件路径,这里测试选择wav
    ffmpeg_path :  ffmpeg.exe的文件路径,ffmpeg是音频转换重要文件,
                   下载官网在http://www.ffmpeg.org/download.html
                   windows的exe下载在这里https://github.com/BtbN/FFmpeg-Builds/releases
##########################################################
'''

if __name__ == '__main__':
    # source_audio = 'D:/study/gra_3_sec/嵌入式系统实验/实验资料/example.m4a'
    # change_audio = 'D:/study/gra_3_sec/嵌入式系统实验/实验资料/example.wav'
    source_audio = input('输入需要更换格式的音频文件:')
    change_audio = input('输入更换格式后的音频文件  :')
    #ffmpeg_path = 'D:/python/ffmpeg/ffmpeg-N-101466-gb7bf631f91-win64-gpl-shared-vulkan/ffmpeg-N-101466-gb7bf631f91-win64-gpl-shared-vulkan/bin/ffmpeg.exe'

    # 执行文件转换的指令
    command = 'ffmpeg' + ' -i' + ' ' + source_audio + ' ' + change_audio

    # 运行指令
    t = Thread(target=execute_cmd_command, args=(command,))
    t.start()


