import random
import os
import shutil
try:
    import gdal
except:
    from osgeo import gdal

def data_clean(img_path, idx_path,  sign):
    """
    :param img_path: 要进行数据清洗的图像数据集的路径（去除无标签图像）。
    :param idx_path: 要进行数据清洗的标签数据集的路径。
    :param sign:  无标签数据的符号表示。(整型)
    :return:
    """
    idx_files = os.listdir(idx_path)
    img_files = os.listdir(img_path)
    for i in idx_files:
        if i[-3:] != 'enp':  #清除金字塔文件(.enp文件)
            idx_dir = os.path.join(idx_path, i)
            idx_file = gdal.Open(idx_dir)
            idx_file = idx_file.ReadAsArray()
            if idx_file == sign:
                os.remove(idx_dir)
                img_dir = os.path.join(img_path, i)
                os.remove(img_dir)
        else:
            os.remove(os.path.join(idx_path,i))

    return 0


def movefile(file_img_dir, file_idx_dir, tar_img_dir, tar_idx_dir, rate):
    """
    :param file_img_dir: 图像的原始路径
    :param file_idx_dir: 标签的原始路径
    :param tar_img_dir:  图像的新路径
    :param tar_idx_dir:  标签的新路径
    :param rate:    自定义抽取的文件比例。比方说100张抽10张，那就是rate = 0.1
    :return: 
    适用于图像与其对应标签的文件名相同的情况。
    """
    files_dir = os.listdir(file_img_dir)
    files_num = len(files_dir)
    pick_num = int(files_num * rate)
    sampler = random.sample(files_dir, pick_num)
    print(sampler)
    for name in sampler:
        shutil.move(file_img_dir + name, tar_img_dir + name)
        shutil.move(file_idx_dir + name, tar_idx_dir + name)
    return 0

if __name__ == "__main__":

    #进行数据清洗，去掉无标签数据。
    img_path = './train_data/images'   #原始图像数据存放路径
    idx_path = './train_data/labels'   #原始标签数据存放路径
    sign = 255   #无标签数据的符号表示
    data_clean(img_path, idx_path, sign)

    #先划分训练集和测试集，再从测试集中抽取出部分数据作为验证集。图像和标签同时划分。
    #划分训练集和测试集
    file_img_dir = './train_data/images/'
    file_idx_dir = './train_data/labels/'
    tar_img_dir = './test_data/images/'
    tar_idx_dir = './test_data/labels/'
    rate = 0.2
    movefile(file_img_dir, file_idx_dir, tar_img_dir, tar_idx_dir, rate)
    test_data_num = len(os.listdir(tar_img_dir))   #获取测试数据集的样本数

    #从上一步得到的训练集中抽取出验证集
    file_img_dir = './train_data/images/'
    file_idx_dir = './train_data/labels/'
    tar_img_dir = './validation_data/images/'
    tar_idx_dir = './validation_data/labels/'
    rate = 0.25
    movefile(file_img_dir, file_idx_dir, tar_img_dir, tar_idx_dir, rate)

    train_data_num = len(os.listdir(file_img_dir))    #获取训练数据集的样本数
    validation_data_num = len(os.listdir(tar_img_dir))    #获取验证数据集的样本数


    print("原始训练集的样本数： ", train_data_num)
    print("原始测试集的样本数： ", test_data_num)
    print("原始验证集的样本数： ", validation_data_num)




