import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the data
trainData = pd.read_csv('project_train.csv')
testData = pd.read_csv('project_test.csv')

# Preprocess the data
X = trainData.drop('Label', axis=1)
y = trainData['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=70)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
x_testFINAL = scaler.transform(testData)

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000,
                                                random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=1000,
                                                min_samples_split = 8,
                                                min_samples_leaf = 2,
                                                random_state = 42),
    'SVM': SVC(probability=True,
               max_iter=1000,
               random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(11, 6),
                        activation='relu',
                        solver='adam',
                        max_iter=1000,
                        random_state=42),
    'LDA': LinearDiscriminantAnalysis(),
    'KNN': KNeighborsClassifier(n_neighbors=7)
    }

# Evaluate classifiers
resultsTrain = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_predTrain= clf.predict(X_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    resultsTrain[name] = {
        'Accuracy': accuracy_score(y_test,y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba),
        'Prediction': clf.predict(x_testFINAL),
        'AccuracyOnTraining': accuracy_score(y_train, y_predTrain),
    }

# Print resultsTrain
for name, metrics in resultsTrain.items():
    print(f"{name}:")
    for metric, value in metrics.items():
        print(metric," :", value)
    print()