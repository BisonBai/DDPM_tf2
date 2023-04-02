import subprocess
import os
ori_input_file = "/home/lqm/Diffusion-Bison/data_long_train/"
ori_output_file = "/home/lqm/Diffusion-Bison/data_long_16/"

files = []
for f in os.listdir(ori_input_file):
    files.append(f)
for f in files:    
    subprocess.call(["ffmpeg", "-i", ori_input_file+f, "-acodec", "pcm_s16le", ori_output_file+f])