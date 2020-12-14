import pycaret
import pandas as pd
import os
import  matplotlib.pyplot as plt
import numpy as np

#import the dataset from pycaret repository
#from pycaret.datasets import get_data
#anomaly = get_data('anomaly')
#import anomaly detection module
from pycaret.anomaly import *
#intialize the setup
#exp_ano = setup(anomaly)




class anamolyDetection:
    def __init__(self):
        self.filePath="/home/k/PycharmProjects/pythonProject/kerasOCR/siemensGomesa/results//"
        self.fileName="df_6.csv"

    def readData(self):
        df1=pd.read_csv(self.filePath+self.fileName)
        print("\n\t shape:",df1.shape)

        return df1

    def createModel(self,df1):

        df1 = setup(df1)
        iforest = create_model('iforest')
        print("\n\t iforest =:",iforest)
        ## plotting a model
        plot_model(iforest)


if __name__=="__main__":

    obj=anamolyDetection()
    df1=obj.readData()







