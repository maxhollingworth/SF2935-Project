import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
#from sklearn.datasets import make_classification

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

def randomForest(testData,testLabel,trainData,trainLabel):
    #clf = RandomForestClassifier()
    clf = RandomForestClassifier(n_estimators=100,
            criterion = 'gini',
            max_depth = None,
            min_samples_split = 2,
            min_samples_leaf = 1,
            min_weight_fraction_leaf = 0.0,
            max_features = 'sqrt',
            max_leaf_nodes = None,
            min_impurity_decrease = 0.0,
            bootstrap = True,
            oob_score = False,
            n_jobs = None,
            random_state = None,
            verbose = 0,
            warm_start = True,
            class_weight = None,
            ccp_alpha = 0.0,
            max_samples = None,
            monotonic_cst = None)

    clf.fit(trainData, trainLabel)
    #print("Test Accuracy: ", clf.score(testData, testLabel))
    #print("Train Accuracy: ", clf.score(trainData, trainLabel))
    testAcc=clf.score(testData,testLabel)
    trainAcc=clf.score(trainData,trainLabel)
    return testAcc, trainAcc

def logisticRegression(testData, testLabel, trainData, trainLabel):
    clf = LogisticRegression(penalty='l2',
                             dual=False,
                             tol=0.0001,
                             C=1.0,
                             fit_intercept=True,
                             intercept_scaling=1,
                             class_weight=None,
                             random_state=None,
                             solver='lbfgs',
                             max_iter=1000,
                             multi_class='deprecated',
                             verbose=0,
                             warm_start=False,
                             n_jobs=None,
                             l1_ratio=None)
    clf.fit(trainData, trainLabel)
    testAcc=clf.score(testData,testLabel)
    trainAcc=clf.score(trainData,trainLabel)
    return testAcc, trainAcc

def neuralNetworkClassifier(testData, testLabel, trainData, trainLabel):
    clf = MLPClassifier(hidden_layer_sizes=(10,),
                        activation='relu',
                        solver='adam',
                        alpha=0.0001,
                        batch_size='auto',
                        learning_rate='constant',
                        learning_rate_init=0.001,
                        power_t=0.5,
                        max_iter=500,
                        shuffle=True,
                        random_state=None,
                        tol=0.0001,
                        verbose=False,
                        warm_start=False,
                        momentum=0.9,
                        nesterovs_momentum=True,
                        early_stopping=False,
                        validation_fraction=0.1,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-08,
                        n_iter_no_change=100,
                        max_fun=15000)
    clf.fit(trainData, trainLabel)
    testAcc=clf.score(testData,testLabel)
    trainAcc=clf.score(trainData,trainLabel)
    return testAcc, trainAcc


if __name__ == '__main__':
    data=readCSV()
    testAccVec=np.array([])
    trainAccVec=np.array([])
    #For running several times just to get some sort of "average" accuracy
    for i in range(0,100):
        testData, testLabel, trainData, trainLabel = splittingData(data)
        testAcc,trainAcc=randomForest(testData, testLabel, trainData, trainLabel)
        testAccVec=np.append(testAccVec,testAcc)
        trainAccVec=np.append(trainAccVec,trainAcc)
    print("Test Accuracy RnF: ",np.mean(testAccVec))
    print("Train Accuracy RnF: ",np.mean(trainAccVec))

    testAccVec=np.array([])
    trainAccVec=np.array([])
    #For running several times just to get some sort of "average" accuracy
    for i in range(0,100):
        testData, testLabel, trainData, trainLabel = splittingData(data)
        testAcc,trainAcc=logisticRegression(testData, testLabel, trainData, trainLabel)
        testAccVec=np.append(testAccVec,testAcc)
        trainAccVec=np.append(trainAccVec,trainAcc)
    print("Test Accuracy LR: ",np.mean(testAccVec))
    print("Train Accuracy LR: ",np.mean(trainAccVec))

    testAccVec=np.array([])
    trainAccVec=np.array([])
    #For running several times just to get some sort of "average" accuracy
    for i in range(0,1):
        testData, testLabel, trainData, trainLabel = splittingData(data)
        testAcc,trainAcc=neuralNetworkClassifier(testData, testLabel, trainData, trainLabel)
        testAccVec=np.append(testAccVec,testAcc)
        trainAccVec=np.append(trainAccVec,trainAcc)
    print("Test Accuracy NN: ",np.mean(testAccVec))
    print("Train Accuracy NN: ",np.mean(trainAccVec))