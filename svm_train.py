import cv2
from imutils import paths
import numpy as np
import os
from sklearn.svm import LinearSVC
import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

# Get the training classes names and store them in a list
train_path = "CPS803 Data/train/"
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = list(paths.list_images(dir))
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# 创建SIFT特征提取器
sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)

# 特征提取与描述子生成
des_list = []

i = 0
for image_path in image_paths:
    i +=1
    im = cv2.imread(image_path)
    im = cv2.resize(im, (300, 300))
    kpts = sift.detect(im)
    kpts, des = sift.compute(im, kpts)
    # print(f"kpts: {kpts[0].shape}")
    print(f"des: {des.shape}")
    des_list.append((image_path, des))
    if i == 10:
        break
    # print("image file path : ", image_path)
    if i%10 == 0:
        print(f"{i} " , end="")
        
# 描述子向量
# print("\ndescriptor")
# i = 0
# descriptors = des_list[0][1]
# for image_path, descriptor in des_list[1:]:
#     descriptors = np.vstack((descriptors, descriptor))
#     i += 1
#     if i%10 == 0:
#         print(". ", end="")

# # 100 聚类 K-Means
# k = 10
# voc, variance = kmeans(descriptors, k, 1)

# # 生成特征直方图
# print("\nfeatures")
# im_features = np.zeros((len(image_paths), k), "float32")
# for i in range(len(image_paths)):
#     words, distance = vq(des_list[i][1], voc)
#     for w in words:
#         im_features[i][w] += 1
    

# # 实现动词词频与出现频率统计
# nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
# idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

# # 尺度化
# print("\nfitting")
# stdSlr = StandardScaler().fit(im_features)
# im_features = stdSlr.transform(im_features)

# # Train the Linear SVM
# clf = LinearSVC()
# clf.fit(im_features, np.array(image_classes))

# # Save the SVM
# print("training and save model...")
# joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)