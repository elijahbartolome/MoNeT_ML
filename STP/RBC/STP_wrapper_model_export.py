import STP_constants as cst
from sys import argv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import joblib
from contextlib import redirect_stdout
import rfpimp as rfp
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from os import path

def wrapperSetup(metric, dataset, path_arg):
    MTR = metric # 'CPT'
    (VT_SPLIT, TREES, DEPTH, KFOLD, JOB) = (
        cst.VT_SPLIT, cst.TREES, cst.DEPTH, cst.KFOLD, cst.JOB
    )

    ###############################################################################
    # Read CSV
    ###############################################################################
    if dataset == "REG":
        DATA = pd.read_csv(path.join(path_arg, 'REG_HLT_50Q_10T.csv'))
    elif dataset == "CLS": 
        DATA = pd.read_csv(path.join(path_arg, 'CLS_HLT_50Q_10T.csv'))
    elif dataset == "SCA":
        DATA = pd.read_csv(path.join(path_arg, 'A_SCA_HLT_50Q_10T.csv'))
    # Features and labels ---------------------------------------------------------
    COLS = list(DATA.columns)
    (FEATS, LABLS) = (
        [i for i in COLS if i[0]=='i'],
        [i for i in COLS if i[0]!='i']
    )

    ###############################################################################
    # Pre-Analysis
    ###############################################################################
    correlation = DATA.corr(method='spearman')
    (f, ax) = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation, mask=np.zeros_like(correlation, dtype=np.bool), 
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True, ax=ax
    )
    # f.show()

    ###############################################################################
    # Split Train/Test
    ###############################################################################
    (inputs, outputs) = (DATA[FEATS], DATA[MTR])
    
    #normalize features
    scaler = preprocessing.MinMaxScaler()
    inputs = pd.DataFrame(scaler.fit_transform(inputs), columns=FEATS)

    (TRN_X, VAL_X, TRN_Y, VAL_Y) = train_test_split(
        inputs, outputs, 
        test_size=float(VT_SPLIT)
    )
    (TRN_L, VAL_L) = [i.shape[0] for i in (TRN_X, VAL_X)]

    return [MTR, VT_SPLIT, TREES, DEPTH, KFOLD, JOB, DATA, FEATS, 
    LABLS, inputs, outputs, TRN_X, VAL_X, TRN_Y, VAL_Y, TRN_L, VAL_L, correlation]

def wrapperTrain(clf, model, TRN_X, TRN_Y, KFOLD, VAL_X, VAL_Y, MTR, FEATS, TRN_L, VAL_L, LABLS, correlation, dataset, path_arg):
    # Final training --------------------------------------------------------------
    clf.fit(TRN_X, TRN_Y)
    joblib.dump(clf, path.join(path_arg, dataset + '_' + model + '_' + MTR + '.pkl'))