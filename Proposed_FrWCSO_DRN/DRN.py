# import keras
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
tf.get_logger().setLevel(logging.ERROR)
import warnings
from tensorflow.keras.layers import Dense
warnings.filterwarnings('ignore')
import random,os, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
import sys, math
from tensorflow.python.util import deprecation
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
deprecation._PRINT_DEPRECATION_WARNINGS = False
from Proposed_FrWCSO_DRN import FrWCSO

def rsnt_bp(x_train, y_train, x_test, y_test, ln):
    x_train=np.tile(x_train, (1, 5))
    a,b = 32,32 # (size of array)
    x_train = np.asarray(x_train[:,:a*b])
    x_train = x_train.reshape((-1,a,b,1))
    x_test = np.tile(x_test, (1, 5))
    x_test = np.asarray(x_test[:, :a*b])
    x_test = x_test.reshape((-1, a, b, 1))
    model = tf.keras.Sequential(
        [
            ResNet50(input_shape=x_train.shape[1:], weights=None, include_top=False, pooling='avg'),
            Dense(ln)
        ]
    )

    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    weight = np.array(model.get_weights())
    model.set_weights(weight + FrWCSO.algm())  # update weight
    model.fit(x_train, y_train, epochs=1, batch_size=20, verbose=0)
    y_pred = model.predict_classes(x_test)
    predict = np.concatenate((y_train,y_pred))
    target = np.concatenate((y_train,y_test))
    return predict,target


def classify(data,lab,tr,A,TPR,TNR,dts):

    from imblearn.over_sampling import  SMOTE as sm
    final_feat,final_label = sm().fit_resample(data,lab)
    tr=tr/100
    x_train, x_test, y_train, y_test = train_test_split(final_feat, final_label, train_size=tr)
    y_train = np.asarray(y_train)
    ln = 2 # n class
    predict,target = rsnt_bp(x_train,y_train,x_test,y_test,ln)
    uni = np.unique(y_test)
    tp, tn, fn, fp = 0, 0, 0, 0
    for i1 in range(len(uni)):
        c = uni[i1]
        for i in range(len(target)):
            if (target[i] == c and predict[i] == c):
                tp = tp + 1
            if (target[i] != c and predict[i] != c):
                tn = tn + 1
            if (target[i] == c and predict[i] != c):
                fn = fn + 1
            if (target[i] != c and predict[i] == c):
                fp = fp + 1
    tn = tn / len(uni)

    A.append((tp + tn) / (tp + tn + fp + fn))
    TPR.append(tp / (tp + fn))
    TNR.append(tn / (tn + fp))
