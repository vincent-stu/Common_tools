import os
import time

start_time = time.time()
txts_folder = r"F:\三维重建\GF7_DLC_E116.5_N40.0_20191226_L1A0000195139\DSM\try"
txts_filenames = os.listdir(txts_folder)
print(txts_filenames)

file = open(r"F:\三维重建\GF7_DLC_E116.5_N40.0_20191226_L1A0000195139\DSM\merge4.txt", 'w')

for filename in txts_filenames:
    file_path = os.path.join(txts_folder, filename)
    #遍历单个文件，读取行数
    for line in open(file_path):
        file.writelines(line)
file.close()

end_time = time.time()
print("融合程序共用时: {}min".format((end_time-start_time)/60))