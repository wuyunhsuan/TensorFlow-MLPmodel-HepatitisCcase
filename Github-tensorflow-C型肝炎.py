"""
reference
Kaggle--Hepatitis C Prediction Dataset
https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

df=pd.read_excel("HepatitisCdata.xlsx")

# convert string labels into numbers
df["category"]=preprocessing.LabelEncoder().fit_transform(df["Category"])
df["sex"]=preprocessing.LabelEncoder().fit_transform(df["Sex"])
# replace N/A with median
df=df.fillna(df.median())

# feature(X), label(Y)
train_x_col=['sex','Age', 'ALB', 'ALP', 'ALT', 'AST','BIL',
             'CHE','CHOL', 'CREA', 'GGT', 'PROT']
labels=['Blood Donor','suspect Blood Donor','Hepatitis',
        'Fibrosis','Cirrhosis']
X=df[train_x_col]
Y=df['category']
print(" X shape",X.shape)
print(" Y shape",Y.shape)

# Transform features to a given range.
scaler = MinMaxScaler()
scaler.fit(X)
X1 = scaler.transform(X)

category=5
dim=12
# split test data to train set and test set
train_x, test_x, train_y, test_y = train_test_split(X1, Y, test_size=0.1)
# one hot encoding
train_y2=tf.keras.utils.to_categorical(train_y, num_classes=category)
test_y2=tf.keras.utils.to_categorical(test_y, num_classes=category)

# build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu,input_dim=dim))
model.add(tf.keras.layers.Dense(units=10*10,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10*10,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10*10,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10*10,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10*10,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=category,activation=tf.nn.softmax))
model.compile(optimizer="adam",
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit(train_x, train_y2,epochs=200,batch_size=128)

# test
score = model.evaluate(test_x, test_y2, batch_size=128)
print("loss:",score[0],"  ","accuracy:",score[1])
predict = model.predict(test_x)
print("First Ans:",np.argmax(predict[0]),"Category:",labels[np.argmax(predict[0])])
print("Second Ans:",np.argmax(predict[1]),"Category:",labels[np.argmax(predict[1])])
print("Third Ans:",np.argmax(predict[2]),"Category:",labels[np.argmax(predict[2])])