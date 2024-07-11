import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder

#########################################
# Read data
TRAIN_PATH = "data/train_master.csv"
x_train, y_train = [], []

train_data = pd.read_csv(TRAIN_PATH)
train_path = train_data['id']
# X_train
for i in range(len(train_path)):
    IMG_PATH = "D:/AI Challenge/Emotion-cls/data/train/train/" + str(train_path[i])
    x_train.append(cv2.imread(IMG_PATH))
x_train = np.array(x_train)

# One hot encoder
onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(train_data[['expression']])
#########################################
np.save('x_train',x_train)
np.save('y_train',y_train)