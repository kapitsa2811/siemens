import pycaret
import pandas as pd
import os
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pycaret.anomaly import *
from collections import defaultdict

import numpy as np
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
from time import time




filePath="/home/k/PycharmProjects/pythonProject/kerasOCR/siemensGomesa/results//"
modelPath="/home/k/PycharmProjects/pythonProject/kerasOCR/siemensGomesa/model//"
#fileName="df_6.csv"
fileName="df1.csv"


def readData():
    df1 = pd.read_csv(filePath + fileName)
    print("\n\t shape:", df1.shape)
    return df1

df1=readData()

train, test = train_test_split(df1, test_size=0.05, shuffle=False)
