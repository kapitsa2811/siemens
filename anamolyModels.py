'''
    reference link: http://www.pycaret.org/tutorials/html/ANO101.html
    This code contains anamoly based model basd on ML

'''


import pycaret
import pandas as pd
import os
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pycaret.anomaly import *
from collections import defaultdict

filePath="/home/k/PycharmProjects/pythonProject/kerasOCR/siemensGomesa/results//"
modelPath="/home/k/PycharmProjects/pythonProject/kerasOCR/siemensGomesa/model//"
#fileName="df_6.csv"
fileName="df1.csv"


def readData():
    df1 = pd.read_csv(filePath + fileName)
    print("\n\t shape:", df1.shape)
    return df1

def count(temp):
    for indx, row in temp.iterrows():
        if row["Anomaly"]==1:
            key=str([row["timeUTC"],row["wind_speed"],row["wind_direction"]])
            #row = tuple(row["timeUTC"],row["wind_speed"],row["wind_direction"])
            #print("\n\t key:",key)
            dAnomoly[key] += 1


df1=readData()

dAnomoly=defaultdict(int)
train, test = train_test_split(df1, test_size=0.05, shuffle=False)

#iforest=createModel()  

train = setup(train, normalize=True, ignore_features=['timeUTC'], session_id=124)  #
#print("\n\t train:",len(train),"\t test:",len(test))
#print("\n\t df type:",train)

'''
    train part isolation forest
'''
iforest = create_model('iforest')
outlier_results_iforest=assign_model(iforest)
outlier_results_iforest.to_csv(filePath+"outlier_results_iforest_train.csv")
save_model(iforest,modelPath+'iforest')
'''
saved_iforest = load_model(modelPath+'iforest')
new_prediction = predict_model(saved_iforest, data=data_unseen)
'''

'''
    prediction part
'''

unseen_predictions = predict_model(iforest, data=test)
#unseen_predictions.head()
unseen_predictions.to_csv(filePath+"outlier_results_iforest_test_predictions.csv")

count(outlier_results_iforest)
#count(unseen_predictions)
#print("\n\t dAnomoly:",dAnomoly)

# dictFrame = pd.DataFrame.from_dict(dAnomoly.items())
# dictFrame.to_csv(filePath+"dictionary.csv")

#####################################################################
iknn = create_model('knn')
outlier_results_iknn=assign_model(iknn)
outlier_results_iknn.to_csv(filePath+"outlier_results_iknn.csv")
save_model(iforest,modelPath+"knn")

'''
    prediction part
'''
unseen_predictions = predict_model(iknn,data=test)
unseen_predictions.to_csv(filePath+"outlier_results_iknn_test_predictions.csv")
count(outlier_results_iknn)
#count(unseen_predictions)
#####################################################################
icluster = create_model('cluster')
outlier_results_icluster=assign_model(icluster)
outlier_results_icluster.to_csv(filePath+"outlier_results_icluster.csv")
save_model(iforest,modelPath+"cluster")
'''
    prediction part
'''

unseen_predictions = predict_model(icluster, data=test)
unseen_predictions.to_csv(filePath+"outlier_results_icluster_test_predictions.csv")
count(outlier_results_icluster)
#count(unseen_predictions)

#####################################################################
iabod = create_model('abod')
outlier_results_abod=assign_model(iabod)
outlier_results_abod.to_csv(filePath+"outlier_results_abod.csv")
save_model(iforest,modelPath+"abod")
'''
    prediction part
'''

unseen_predictions = predict_model(iabod, data=test)
unseen_predictions.to_csv(filePath+"outlier_results_iabod_test_predictions.csv")
count(outlier_results_abod)
#count(unseen_predictions)

#####################################################################
ihistogram = create_model('histogram')
outlier_results_ihistogram=assign_model(ihistogram)
outlier_results_ihistogram.to_csv(filePath+"outlier_results_ihistogram.csv")
save_model(iforest,modelPath+"histogram")
'''
    prediction part
'''

unseen_predictions = predict_model(ihistogram, data=test)
unseen_predictions.to_csv(filePath+"outlier_results_ihistogram_test_predictions.csv")
count(outlier_results_ihistogram)
#count(unseen_predictions)

#####################################################################

ilof = create_model('lof')
outlier_results_ilof=assign_model(ilof)
outlier_results_ilof.to_csv(filePath+"outlier_results_ilof.csv")
save_model(iforest,modelPath+"lof")
'''
    prediction part
'''

unseen_predictions = predict_model(ilof, data=test)
unseen_predictions.to_csv(filePath+"outlier_results_ilof_test_predictions.csv")
count(outlier_results_ilof)
#count(unseen_predictions)

#####################################################################

ipca = create_model('pca')
outlier_results_ipca=assign_model(ipca)
outlier_results_ipca.to_csv(filePath+"outlier_results_ipca.csv")
save_model(iforest,modelPath+"pca")
'''
    prediction part
'''

unseen_predictions = predict_model(ipca, data=test)
#unseen_predictions.to_csv(filePath+"outlier_results_ipca_test_predictions.csv")

count(outlier_results_ipca)
#count(unseen_predictions)

#####################################################################
#####################################################################

'''
icof = create_model('cof')
outlier_results_icof=assign_model(icof)
outlier_results_icof.to_csv(filePath+"outlier_results_icof.csv")
save_model(icof,modelPath+"cof")
'''
    #prediction part
'''

unseen_predictions = predict_model(icof, data=test)
#unseen_predictions.to_csv(filePath+"outlier_results_ipca_test_predictions.csv")

count(outlier_results_icof)
#count(unseen_predictions)
'''
#####################################################################

#####################################################################

isvm = create_model('svm')
outlier_results_isvm=assign_model(isvm)
outlier_results_isvm.to_csv(filePath+"outlier_results_isvm.csv")
save_model(isvm,modelPath+"svm")
'''
    prediction part
'''

unseen_predictions = predict_model(isvm, data=test)
#unseen_predictions.to_csv(filePath+"outlier_results_ipca_test_predictions.csv")

count(outlier_results_isvm)
#count(unseen_predictions)

#####################################################################

#####################################################################

imcd = create_model('mcd')
outlier_results_imcd=assign_model(imcd)
outlier_results_imcd.to_csv(filePath+"outlier_results_imcd.csv")
save_model(imcd,modelPath+"mcd")
'''
    prediction part
'''

unseen_predictions = predict_model(imcd, data=test)
#unseen_predictions.to_csv(filePath+"outlier_results_ipca_test_predictions.csv")

count(outlier_results_imcd)
#count(unseen_predictions)

#####################################################################
# disabling
#####################################################################

# isod = create_model('sod')
# outlier_results_isod=assign_model(isod)
# outlier_results_isod.to_csv(filePath+"outlier_results_isod.csv")
# save_model(isod,modelPath+"sod")
# '''
#     prediction part
# '''
#
# unseen_predictions = predict_model(isod, data=test)
# #unseen_predictions.to_csv(filePath+"outlier_results_ipca_test_predictions.csv")
#
# count(outlier_results_isod)
# #count(unseen_predictions)

#####################################################################

#####################################################################

# isos = create_model('sos')
# outlier_results_isos=assign_model(isos)
# outlier_results_isos.to_csv(filePath+"outlier_results_isos.csv")
# save_model(isos,modelPath+"sos")
# '''
#     prediction part
# '''
#
# unseen_predictions = predict_model(isos, data=test)
# #unseen_predictions.to_csv(filePath+"outlier_results_ipca_test_predictions.csv")
#
# count(outlier_results_isos)
# #count(unseen_predictions)
#
#####################################################################


#dictFrame1=pd.DataFrame(columns=[])
dictFrame = pd.DataFrame.from_dict(dAnomoly.items())
dictFrame.to_csv(filePath+"dictionary.csv")

