import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras import backend as K
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from sklearn.ensemble import VotingClassifier
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import h5py
import pickle

data = pd.read_csv("test.csv")
X = data.drop("fraud", axis = 1)
y = data["fraud"]
y_num = y.values
with open("knn.pkl", "rb") as f:
	knn = pickle.load(f)
lstm_output_size = 70
data_combo = pd.read_csv("train.csv")
X_combo = data_combo.drop("fraud", axis = 1)
y_combo = data_combo["fraud"]
test_labels_combo = np.array(y_combo)
test_features_combo = np.reshape(X_combo.values, (X_combo.values.shape[0],X_combo.values.shape[1],1))
cnn = Sequential()
cnn.add(Convolution1D(64, 3, activation="relu", input_shape= (8, 1)))
cnn.add(Convolution1D(64, 3, activation="relu"))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Convolution1D(128, 3, activation="relu", padding = "same"))
cnn.add(Convolution1D(128, 3, activation="relu", padding = "same"))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(LSTM(lstm_output_size))
cnn.add(Dropout(0.1))
cnn.add(Dense(2, activation="softmax"))
cnn.load_weights("cnn_model.hdf5")
cnn.compile(loss="sparse_categorical_crossentropy", optimizer="SGD", metrics=['accuracy'])
print("Created model and loaded weights from file")
X_combo['fraud'] = cnn.predict_classes(test_features_combo)
K.clear_session()
full_data = X_combo
full_labels = full_data['fraud']
full_features = full_data.drop('fraud', axis = 1)
full_features_array = full_features.values
full_labels_array = full_labels.values
train_features,test_features,train_labels,test_labels=train_test_split(
full_features_array,full_labels_array,train_size=0.80,test_size=0.20)
train_features=normalize(train_features)
test_features=normalize(test_features)
combo=KNeighborsClassifier(n_neighbors=4,algorithm="kd_tree",n_jobs=-1)
combo.fit(train_features,train_labels.ravel())
# combo_predicted_test_labels=combo.predict(test_features)
# tn,fp,fn,tp=confusion_matrix(test_labels,combo_predicted_test_labels).ravel()
# combo_accuracy_score=accuracy_score(test_labels,combo_predicted_test_labels)
# combo_precison_score=precision_score(test_labels,combo_predicted_test_labels)
# combo_recall_score=recall_score(test_labels,combo_predicted_test_labels)
# combo_f1_score=f1_score(test_labels,combo_predicted_test_labels)
def cnnmodel_test( a_knn, b_knn, c_knn, d_knn, e_knn, f_knn, g_knn, h_knn ):
    arr_testing=pd.DataFrame({'customer':[a_knn],'age':[b_knn],'gender':[c_knn],'zipcodeOri':[d_knn],'merchant':[e_knn],'zipMerchant':[f_knn],'category':[g_knn],'amount':[h_knn]},columns=['customer','age','gender','zipcodeOri','merchant','zipMerchant','category','amount'])
    test_features1 = np.reshape(arr_testing.values, (arr_testing.values.shape[0],arr_testing.values.shape[1],1))
    cnn = Sequential()
    cnn.add(Convolution1D(64, 3, activation="relu", input_shape= (8, 1)))
    cnn.add(Convolution1D(64, 3, activation="relu"))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(Convolution1D(128, 3, activation="relu", padding = "same"))
    cnn.add(Convolution1D(128, 3, activation="relu", padding = "same"))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(LSTM(lstm_output_size))
    cnn.add(Dropout(0.1))
    cnn.add(Dense(2, activation="softmax"))
    cnn.load_weights("cnn_model.hdf5")
    cnn.compile(loss="sparse_categorical_crossentropy", optimizer="SGD", metrics=['accuracy'])
    print("Created model and loaded weights from file")
    y_cnn = cnn.predict_classes(test_features1)
    K.clear_session()
    return str(y_cnn[0])
def knnmodel_test( a_knn, b_knn, c_knn, d_knn, e_knn, f_knn, g_knn, h_knn ):
    arr_testing=pd.DataFrame({'customer':[a_knn],'age':[b_knn],'gender':[c_knn],'zipcodeOri':[d_knn],'merchant':[e_knn],'zipMerchant':[f_knn],'category':[g_knn],'amount':[h_knn]},columns=['customer','age','gender','zipcodeOri','merchant','zipMerchant','category','amount'])
    y_knn = knn.predict(arr_testing)
    return str(y_knn[0])

def knnmodel_accuracy():
    y_knn = knn.predict(test_features)
    return accuracy_score(test_labels,y_knn)
def cnnmodel_accuracy():
    # test_labels = np.array(y)
    test_features1 = np.reshape(test_features, (test_features.shape[0],test_features.shape[1],1))
    cnn = Sequential()
    cnn.add(Convolution1D(64, 3, activation="relu", input_shape= (8, 1)))
    cnn.add(Convolution1D(64, 3, activation="relu"))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(Convolution1D(128, 3, activation="relu", padding = "same"))
    cnn.add(Convolution1D(128, 3, activation="relu", padding = "same"))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(LSTM(lstm_output_size))
    cnn.add(Dropout(0.1))
    cnn.add(Dense(2, activation="softmax"))
    cnn.load_weights("cnn_model.hdf5")
    cnn.compile(loss="sparse_categorical_crossentropy", optimizer="SGD", metrics=['accuracy'])
    print("Created model and loaded weights from file")
    y_cnn = cnn.predict_classes(test_features1)
    K.clear_session()
    return accuracy_score(test_labels,y_cnn)
def combomodel_test( a_knn, b_knn, c_knn, d_knn, e_knn, f_knn, g_knn, h_knn ):
    arr_testing=pd.DataFrame({'customer':[a_knn],'age':[b_knn],'gender':[c_knn],'zipcodeOri':[d_knn],'merchant':[e_knn],'zipMerchant':[f_knn],'category':[g_knn],'amount':[h_knn]},columns=['customer','age','gender','zipcodeOri','merchant','zipMerchant','category','amount'])
    y_combo = combo.predict(arr_testing)
    return str(y_combo[0])

def combomodel_accuracy():
    y_combo = combo.predict(test_features)
    return accuracy_score(test_labels,y_combo)

from flask import Flask, redirect, url_for, request
app = Flask(__name__)

@app.route("/accuracy/")
def hel():
    #knnmodel()
    x = knnmodel_accuracy()
    z = cnnmodel_accuracy()
    cm = combomodel_accuracy()
    y="Accuracy of the KNN model is: " + str(x) + "<br>\n Accuracy of the CNN Model is :" + str(z) + "<br>Combo Accurracy:"+str(cm)
    return str(y)

@app.route("/test/", methods=['GET','POST'])
def hello():
    #knnmodel()
    if request.method == 'POST':
        a_knn = request.form['customer']
        b_knn = request.form['age']
        c_knn = request.form['gender']
        d_knn = request.form['zipcodeOri']
        e_knn = request.form['merchant']
        f_knn = request.form['zipMerchant']         
        g_knn = request.form['category']
        h_knn = request.form['amount']
        y="Result from KNN model: "+ str(knnmodel_test( a_knn, b_knn, c_knn, d_knn, e_knn, f_knn, g_knn, h_knn ))
        y=y +"<br>Result from KNN model: "+ str(cnnmodel_test( a_knn, b_knn, c_knn, d_knn, e_knn, f_knn, g_knn, h_knn ))
        y=y +"<br>Result from Combo model: "+ str(combomodel_test( a_knn, b_knn, c_knn, d_knn, e_knn, f_knn, g_knn, h_knn ))
        return y
    if request.method == 'GET':
    	return "Error Wrong Request"
    return str(-1)


if __name__=='__main__':
	app.run()