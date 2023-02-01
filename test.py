import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from keras.utils import np_utils
from keras import backend as K
from tensorflow.python.framework import ops as tf_ops
import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
import time
seed=42
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
zhang=5000
n=zhang*10
width=28
X = np.zeros((n,1,width,width),dtype = np.float64)
Y = np.zeros((n,),dtype = np.float64)
path = '../train5/'
objects = os.listdir(path)
m = 0
for obj in objects:
    m = m + 1
j = 0
def channel_max(x):
    x=K.max(x,axis=-1,keepdims=True)
    return x
def channel_avg(x):
    x=K.mean(x,axis=-1,keepdims=True)
    return x
def se_block(inputs,ratio=4):
    x_max=Lambda(channel_max)(inputs)
    x_avg=Lambda(channel_avg)(inputs)
    x=Concatenate(axis=2)([x_max,x_avg])
    x=Conv1D(1,kernel_size=7,padding='same',activation='sigmoid')(x)
    return Multiply()([inputs,x])

def channel_block(inputs,ratio=4):
    channel = inputs.shape[-1]

    share_dense1=Dense(channel//4)
    share_dense2=Dense(channel)

    x_max = GlobalMaxPooling1D()(inputs)
    x_avg = GlobalAveragePooling1D()(inputs)

    x_max = Reshape([1, -1])(x_max)
    x_avg = Reshape([1, -1])(x_avg)

    x_max = share_dense1(x_max)
    x_max = share_dense2(x_max)

    x_avg = share_dense1(x_avg)
    x_avg = share_dense2(x_avg)

    x = Add()([x_max,x_avg])
    x = Activation('sigmoid')(x)

    out = Multiply()([inputs,x])
    return out

for obj in objects:
    for i in tqdm(range(zhang)):
        for k in range(1):
          s=str(i + k)
          #print(obj)
          #rint('train2/' + obj + '/%s.jpeg' % s)
          X[i + j][k] = cv2.cvtColor(cv2.imread(path + obj + '/%s.jpeg' % s), cv2.COLOR_BGR2GRAY)
          break
    j = j + zhang
X = np.expand_dims(X, axis=4)
for i in range(m):
    Y[int((i + 1) * n / m):] =  i + 1
X=X.reshape(-1,784,1)
X/=255
y = np_utils.to_categorical(Y, m)
from sklearn.model_selection import train_test_split
X_train,X_valid,Y_train,Y_valid = train_test_split(X,y,test_size = 0.2)

convs = []
convs2=[]
inputs =Input(shape=(784,1))
conv1_1=Conv1D(name='conv1_1',filters=32,kernel_size=3,activation= 'relu',padding="same")(inputs)
dp1_1=Dropout(0.5)(conv1_1)
conv1_2=Conv1D(name='conv1_2',filters=32,strides=5,kernel_size=3,activation= 'relu',padding="same")(dp1_1)
conv1_3=Conv1D(name='conv1_3',filters=32,kernel_size=3,activation= 'relu',padding="same")(conv1_2)
dp1_3=Dropout(0.5)(conv1_3)
conv1_4=Conv1D(name='conv1_4',filters=32,strides=5,kernel_size=3,activation= 'relu',padding="same")(dp1_3)
#conv1_3=Flatten()(conv1_2)
convs.append(conv1_4)

conv3_1=Conv1D(name='conv3_1',filters=32,kernel_size=5,activation= 'relu',padding="same")(inputs)
dp3_1=Dropout(0.5)(conv3_1)
conv3_2=Conv1D(name='conv3_2',filters=32,strides=5,kernel_size=5,activation= 'relu',padding="same")(dp3_1)
conv3_3=Conv1D(name='conv3_3',filters=32,kernel_size=3,activation= 'relu',padding="same")(conv3_2)
dp3_3=Dropout(0.5)(conv3_3)
conv3_4=Conv1D(name='conv3_4',filters=32,strides=5,kernel_size=3,activation= 'relu',padding="same")(dp3_3)
#conv3_3=Flatten()(conv3_2)
convs.append(conv3_4)

conv4_1=Conv1D(name='conv4_1',filters=32,kernel_size=7,activation= 'relu',padding="same")(inputs)
dp4_1=Dropout(0.5)(conv4_1)
conv4_2=Conv1D(name='conv4_2',filters=32,strides=5,kernel_size=7,activation= 'relu',padding="same")(dp4_1)
conv4_3=Conv1D(name='conv4_3',filters=32,kernel_size=3,activation= 'relu',padding="same")(conv4_2)
dp4_3=Dropout(0.5)(conv4_3)
conv4_4=Conv1D(name='conv4_4',filters=32,strides=5,kernel_size=3,activation= 'relu',padding="same")(dp4_3)
#conv4_3=Flatten()(conv4_2)
convs.append(conv4_4)


merge = keras.layers.concatenate(convs, axis=2)

#attention
# shape=merge.shape[-1]
# x2=Dense(shape,activation='softmax')(merge)
# z2=Multiply()([merge,x2])

z2=channel_block(merge)
z2=se_block(z2)

conv5_1=Conv1D(name='conv5_1',filters=64,kernel_size=3,activation= 'relu',padding="same")(z2)
conv5_2=Conv1D(name='conv5_2',filters=64,kernel_size=3,strides=5,activation= 'relu',padding="same")(conv5_1)
f1=Flatten()(conv5_2)
#print(y)
fc1=Dense(64)(f1)
cnn_feature=Flatten(name='Flatten1')(fc1)


convs2.append(cnn_feature)
inputs2=Embedding(256,32, input_length=784)(inputs)
inputs3=Reshape((784, 32))(inputs2)
conv2_1=Bidirectional(GRU(64,dropout=0.3,return_sequences=True))(inputs3)
conv2_2=Bidirectional(GRU(32,dropout=0.3))(conv2_1)

#gru的attention
shape=conv2_2.shape[-1]
x=Dense(shape,activation='softmax',name="rnnatten")(conv2_2)
z=Multiply()([conv2_2,x])


convs2.append(z)
for conv in convs2:
    print(conv)
merge2 = keras.layers.concatenate(convs2, axis=1)
shape=merge2.shape[-1]

output=Dense(10,activation='softmax')(merge2 )
model =Model(inputs=inputs, outputs=output)

model.summary()
class_weight = {
            0: 1.0,
            1: 1.0,
            2: 5.0,
            3: 1.0,
            4: 1.0,
            5: 1.0,
            6: 1.0,
            7: 1.0,
            8: 5.0,
            9: 1.0,
        }
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.9, epsilon=1e-06, decay=0.0)
model.compile(loss="categorical_crossentropy",optimizer=adam, metrics=['accuracy'])

start = time.time()
h = model.fit(X_train, Y_train,batch_size=64,class_weight =class_weight ,epochs=42,verbose=1,validation_split=0.1)
end = time.time()
T=end-start
print(T)
#model.save('1d_cnn-bigru.h5')

start = time.time()
loss,acc = model.evaluate(X_valid,Y_valid,)
end = time.time()
T=end-start
print('测试时间',T)
print('\ntest loss',loss)
print('test acc',acc)
print(X_valid.shape)
from sklearn.metrics import classification_report, confusion_matrix
import itertools

Y_pred = model.predict(X_valid)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

target_names = objects

print(classification_report(np.argmax(Y_valid, axis=1), y_pred, target_names=target_names,digits=4))

print(confusion_matrix(np.argmax(Y_valid, axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(Y_valid, axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure(figsize=(14, 12))
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix')
plt.savefig('t1.svg', dpi=300)
plt.show()

