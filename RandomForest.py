import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def readCSV():
    #danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,Label
    with open('project_train.csv', 'r',encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        data = list(reader)
        data_array = np.array(data, dtype='float32')
        return data_array
def splittingData(data):
    label=data[:,-1]
    data=np.delete(data,-1,axis=-1)
    trainData, testData, trainLabel, testLabel = train_test_split(data, label, test_size=0.2)
    return testData, testLabel, trainData, trainLabel

if __name__ == '__main__':
    data=readCSV()
    testData, testLabel, trainData, trainLabel = splittingData(data)
    clf=RandomForestClassifier()
    clf.fit(trainData,trainLabel)
    print(clf.score(testData,testLabel))

