# python 遥感图像分类


# 导入库
from osgeo import gdal
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

#设置中文字体
mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 12  # 字体大小
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

start_time = time.time()
# 读取图像数据
raster0 = gdal.Open(r"G:\landsat8_fei\LC08_L1TP_123032_20190902_20190916_01_T1_yanjiuqu.tif")

# 获取空间信息
img_width = raster0.RasterXSize            # 栅格矩阵的列数
img_height = raster0.RasterYSize           # 栅格矩阵的行数
img_bands = raster0.RasterCount            # 波段数
img_proj = raster0.GetProjection()         # 获取投影信息
img_geotrans = raster0.GetGeoTransform()   # 获取仿射矩阵信息

# 图像数据转为列格式
raster0.ReadAsArray().shape
rarray0 = raster0.ReadAsArray().transpose(1,2,0).reshape(round(raster0.ReadAsArray().size/7),7)
print(rarray0)

# 读取训练样本
raster_plants = gdal.Open(r"G:\landsat8_fei\train_data\plants_train.tif")
raster_waters = gdal.Open(r"G:\landsat8_fei\train_data\waters_train.tif")
raster_buildings = gdal.Open(r"G:\landsat8_fei\train_data\buildings_train.tif")
raster_barelands = gdal.Open(r"G:\landsat8_fei\train_data\barelands_train.tif")


# 定义函数显示图像信息
def show_img_imfo(raster):

    # 投影
    print("-----------------------------------------------------------------")
    print("数据投影：", raster.GetProjection)
    print("-----------------------------------------------------------------")
    
    # 行列数目
    print("列数：", raster.RasterXSize)
    print("-----------------------------------------------------------------")
    print("行数：", raster.RasterYSize)
    print("-----------------------------------------------------------------")
    
    # 波段数量
    print("波段数量：", raster.RasterCount)
    print("-----------------------------------------------------------------")
    
    # 元数据
    print("元数据：", raster.GetMetadata)
    print("-----------------------------------------------------------------")

# 调用函数显示图像信息
print("待分类遥感图像信息：")
show_img_imfo(raster0)
print('\n')
print("植被样本图像信息：")
show_img_imfo(raster_plants)
print('\n')
print("水体样本图像信息：")
show_img_imfo(raster_waters)
print('\n')
print("建筑物样本图像信息：")
show_img_imfo(raster_buildings)
print('\n')
print("裸地样本图像信息：")
show_img_imfo(raster_barelands)
print('\n')

# 将训练样本转换成 array，并变为列格式
rarray_plants = raster_plants.ReadAsArray().transpose(1,2,0).reshape(round(raster_plants.ReadAsArray().size/7),7)
rarray_waters = raster_waters.ReadAsArray().transpose(1,2,0).reshape(round(raster_waters.ReadAsArray().size/7),7)
rarray_buildings = raster_buildings.ReadAsArray().transpose(1,2,0).reshape(round(raster_buildings.ReadAsArray().size/7),7)
rarray_barelands = raster_barelands.ReadAsArray().transpose(1,2,0).reshape(round(raster_barelands.ReadAsArray().size/7),7)


# 添加列标签
array_plants = np.c_[rarray_plants, np.ones(rarray_plants.shape[0])*1]
array_waters = np.c_[rarray_waters, np.ones(rarray_waters.shape[0])*2]
array_buildings = np.c_[rarray_buildings, np.ones(rarray_buildings.shape[0])*3]
array_barelands = np.c_[rarray_barelands, np.ones(rarray_barelands.shape[0])*4]


# 纵向合并数据
data_with_0 = np.vstack((array_plants,array_waters,array_buildings,array_barelands))

# 过滤空值，生成训练数据
data = data_with_0[~(np.any(data_with_0 == 0, axis = 1))]

# 可以看出选了 85509 个像素作为训练样本
print('训练样本信息如下：')
print(data.shape)

# 导入 sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

# 划分数据
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    data[:,0:7],             # 像元值
    data[:,7].reshape(len(data[:,7]),1),                # 类别标签
    test_size=0.3,          # 测试集比例
)
# RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=500, random_state=42)

# 拟合模型
rf_clf.fit(Xtrain, Ytrain)

# 预测
some_digit_predict = rf_clf.predict(Xtest)
# 模型精度
OA=sum(some_digit_predict == Ytest.reshape(len(some_digit_predict,)))/len(some_digit_predict)
Kappa = cohen_kappa_score(Ytest, some_digit_predict)
mat = confusion_matrix(Ytest, some_digit_predict)
plt.figure(figsize=(6,6))
sns.heatmap(mat, square = True, annot=True, cmap='Blues',cbar=False)
plt.title('混淆矩阵')
plt.xlabel('预测类别\n OA={:.2f}%; Kappa={:.2f}'.format(OA*100, Kappa))
plt.ylabel('真实类别')
plt.savefig(r'G:\landsat8_fei\分类结果的混淆矩阵_RF5.jpg', dpi = 300)



# 对测试集进行分类
predict1 = rf_clf.predict(rarray0)
predict1  = predict1.reshape(2951,3386)        # 行数 * 列数
# 输出tiff图像
driver = gdal.GetDriverByName("GTiff")
dataset = driver.Create(r"G:\landsat8_fei\classification_result\result_rf5.tif",3386,2951,1,gdal.GDT_Float32)
dataset.SetGeoTransform(img_geotrans)
dataset.SetProjection(img_proj)
dataset.GetRasterBand(1).WriteArray(predict1)
del dataset

end_time = time.time()
print("该程序共用时：{}s".format((end_time-start_time)))
#程序终结标识
print('end')