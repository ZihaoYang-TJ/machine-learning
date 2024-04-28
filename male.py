import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import csv

with open('/work/data/test_data.csv', newline='') as inputFile, open('/work/output.csv', newline='',
                                                                     mode='w') as outputFile:
    fieldnames = ['ID', 'CHANNEL_A', 'CHANNEL_B', 'CHANNEL_C']
    writer = csv.DictWriter(outputFile, fieldnames=fieldnames)
    writer.writeheader()
    inputReader = csv.DictReader(inputFile)

    # print(inputReader.fieldnames)
    for row in inputReader:
        writer.writerow({'ID': row['ID'], 'CHANNEL_A': 1, 'CHANNEL_B': 1, 'CHANNEL_C': 1})


COL = pd.read_csv('/work/data/train_base_data.csv', usecols=[f'COL{i}' for i in range(1, 55)])
mapping = {chr(i): i - ord('A') + 1 for i in range(ord('A'), ord('Z') + 1)}
COL['COL3'] = COL['COL3'].map(mapping)
COL['COL4'] = COL['COL4'].map(mapping)
COL['COL5'] = COL['COL5'].map(mapping)
COL['COL19'] = COL['COL19'].map(mapping)
COL_test = pd.read_csv('/work/data/test_data.csv', usecols=[f'COL{i}' for i in range(1, 55)])
COL_test['COL3'] = COL_test['COL3'].map(mapping)
COL_test['COL4'] = COL_test['COL4'].map(mapping)
COL_test['COL5'] = COL_test['COL5'].map(mapping)
COL_test['COL19'] = COL_test['COL19'].map(mapping)
columns_to_process = ['COL1', 'COL2', 'COL6','COL9','COL10','COL13','COL14','COL15','COL16','COL17','COL18','COL21','COL24','COL25','COL29','COL31','COL34','COL35','COL36','COL37','COL38','COL39','COL40','COL42','COL43','COL45','COL46','COL47','COL50','COL51','COL52']  # 在这里指定要处理的列名
COL[columns_to_process] = COL[columns_to_process].fillna(COL.mean())
# COL[columns_to_process] = COL[columns_to_process].replace(0, COL.mean())
columns_to_process = ['COL3', 'COL4','COL5', 'COL7','COL8','COL11','COL12','COL19','COL20','COL22','COL23','COL26','COL27','COL28','COL30','COL32','COL33','COL41','COL44','COL48','COL49','COL53','COL54']
mode_values = COL[columns_to_process].mode().iloc[0]
COL[columns_to_process] = COL[columns_to_process].fillna(mode_values)
# COL[columns_to_process] = COL[columns_to_process].replace(0, mode_values)
# scaler = StandardScaler()
# COL = pd.DataFrame(scaler.fit_transform(COL), columns=COL.columns)
columns_to_process = ['COL1', 'COL2', 'COL6','COL9','COL10','COL13','COL14','COL15','COL16','COL17','COL18','COL21','COL24','COL25','COL29','COL31','COL34','COL35','COL36','COL37','COL38','COL39','COL40','COL42','COL43','COL45','COL46','COL47','COL50','COL51','COL52']  # 在这里指定要处理的列名
COL_test[columns_to_process] = COL_test[columns_to_process].fillna(COL.mean(numeric_only=True))
# COL_test[columns_to_process] = COL_test[columns_to_process].replace(0, COL.mean(numeric_only=True))
# columns_to_process = ['COL3', 'COL4','COL5', 'COL7','COL8','COL11','COL12','COL19','COL20','COL22','COL23','COL26','COL27','COL28','COL30','COL32','COL33','COL41','COL44','COL48','COL49','COL53','COL54']
# mode_values = COL_test[columns_to_process].mode().iloc[0]
COL_test[columns_to_process] = COL_test[columns_to_process].fillna(mode_values)
# COL_test[columns_to_process] = COL_test[columns_to_process].replace(0, mode_values)
scaler = StandardScaler()
COL_test = pd.DataFrame(scaler.fit_transform(COL_test), columns=COL_test.columns)
from joblib import dump, load
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification

CHANNEL_A = pd.read_csv('/work/data/train_base_data.csv')['CHANNEL_A']
CHANNEL_B = pd.read_csv('/work/data/train_base_data.csv')['CHANNEL_B']
CHANNEL_C = pd.read_csv('/work/data/train_base_data.csv')['CHANNEL_C']

# clf = RandomForestClassifier(class_weight='balanced').fit(COL, CHANNEL_A)
clf = BaggingClassifier().fit(COL, CHANNEL_A)
# dump(clf, 'model_A.joblib')
A = clf.predict_proba(COL_test)
df = pd.read_csv('/work/output.csv')
z = 0
for i in range(len(A)):
    if A[i][0] > 0.7:
        A[i][0] = 0
    else:
        A[i][0] = 1
df['CHANNEL_A'] = A[:, 0].astype(int)
df.to_csv('/work/output.csv', index=False)
# clf = RandomForestClassifier(class_weight='balanced').fit(COL, CHANNEL_B)
clf = BaggingClassifier().fit(COL, CHANNEL_B)
# dump(clf, 'model_B.joblib')
A = clf.predict_proba(COL_test)
df = pd.read_csv('/work/output.csv')
z = 0
for i in range(len(A)):
    if A[i][0] > 0.6:
        A[i][0] = 0
    else:
        A[i][0] = 1
df['CHANNEL_B'] = A[:, 0].astype(int)
df.to_csv('/work/output.csv', index=False)
# clf = RandomForestClassifier(class_weight='balanced').fit(COL, CHANNEL_C)
clf = BaggingClassifier().fit(COL, CHANNEL_C)
A = clf.predict_proba(COL_test)
df = pd.read_csv('/work/output.csv')
z = 0
for i in range(len(A)):
    if A[i][0] > 0.7:
        A[i][0] = 0
    else:
        A[i][0] = 1
df['CHANNEL_C'] = A[:, 0].astype(int)
df.to_csv('/work/output.csv', index=False)