import STP_constants as cst
from sys import argv
from pathlib import Path
import joblib
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics, preprocessing
from contextlib import redirect_stdout
import rfpimp as rfp

(VT_SPLIT, TREES, DEPTH, KFOLD, JOB) = (
        cst.VT_SPLIT, cst.TREES, cst.DEPTH, cst.KFOLD, cst.JOB
    )

if argv[1] == "REG":
    DATA = pd.read_csv(path.join(argv[4], 'REG_HLT_50Q_10T.csv'))
elif argv[1] == "CLS": 
    DATA = pd.read_csv(path.join(argv[4], 'CLS_HLT_50Q_10T.csv'))
elif argv[1] == "SCA":
    DATA = pd.read_csv(path.join(argv[4], 'A_SCA_HLT_50Q_10T.csv'))
# Features and labels ---------------------------------------------------------
COLS = list(DATA.columns)
(FEATS, LABLS) = (
    [i for i in COLS if i[0]=='i'],
    [i for i in COLS if i[0]!='i']
)

correlation = DATA.corr(method='spearman')

(inputs, outputs) = (DATA[FEATS], DATA[argv[3]])

#normalize features
scaler = preprocessing.MinMaxScaler()
inputs = pd.DataFrame(scaler.fit_transform(inputs), columns=FEATS)

(TRN_X, VAL_X, TRN_Y, VAL_Y) = train_test_split(
        inputs, outputs, 
        test_size=float(VT_SPLIT)
    )

(TRN_L, VAL_L) = [i.shape[0] for i in (TRN_X, VAL_X)]

file_name = Path(__file__).parents[0] / "input/{dataset_name}_{model_name}_{metric_name}.pkl".format(dataset_name=argv[1], model_name=argv[2], metric_name=argv[3])
clf = joblib.load(file_name)

kScores = cross_val_score(
        clf, TRN_X, TRN_Y.values.ravel(), 
        cv=int(KFOLD), 
        scoring=metrics.make_scorer(metrics.f1_score, average='weighted')
    )

PRD_Y = clf.predict(VAL_X)

if argv[1] == 'CLS':
    proba_predict = clf.predict_proba(VAL_X)
    (accuracy, f1, precision, recall, jaccard, log) = (
        metrics.accuracy_score(VAL_Y, PRD_Y),
        metrics.f1_score(VAL_Y, PRD_Y, average='weighted'),
        metrics.precision_score(VAL_Y, PRD_Y, average='weighted'),
        metrics.recall_score(VAL_Y, PRD_Y, average='weighted'),
        metrics.jaccard_score(VAL_Y, PRD_Y, average='weighted'),
        metrics.log_loss(VAL_Y, proba_predict)
    )
    report = metrics.classification_report(VAL_Y, PRD_Y)

    try:
        featImportance = list(clf.feature_importances_)
        impDC = rfp.oob_dropcol_importances(clf, TRN_X, TRN_Y.values.ravel())
        impDCD = impDC.to_dict()['Importance']
        impPM = rfp.importances(clf, TRN_X, TRN_Y)
        impPMD = impPM.to_dict()['Importance']
    except AttributeError:
        pass

    with open(path.join(argv[4], argv[1] + '_' + argv[2] + '_' + argv[3] + '.txt'), 'w') as f:
        with redirect_stdout(f):
            print('* Train/Validate entries: `{}/{} ({})`'.format(TRN_L, VAL_L, TRN_L+VAL_L))
            print('* Cross-validation F1: `%0.2f (+/-%0.2f)`'%(kScores.mean(), kScores.std()*2))
            print('* Validation Accuracy: `{:.2f}`'.format(accuracy))
            print('* Validation F1: `{:.2f} ({:.2f}/{:.2f})`'.format(f1, precision, recall))
            print('* Jaccard: `{:.2f}`'.format(jaccard))
            print('* Log Loss: `{:.2f}`'.format(log))
            if 'featImportance' in locals():
                print('* Features Importance & Correlation')
                print('```')
                for i in zip(FEATS, featImportance, correlation[LABLS[0]]):
                    print('{}: {:.3f}, {:.3f}'.format(*i))
                if argv[2] != 'GBT':
                    print('```')
                    print('* Drop-Cols & Permutation Features Importance')
                    print('```')
                    for i in FEATS:
                        print('{}: {:.3f}, {:.3f}'.format(i, impDCD[i], impPMD[i]))
                    print('```')
            print('* Class report: ')
            print('```')
            print(report)
            print('```')

elif argv[1] == 'REG' or argv[1] == 'SCA':
    kScores = cross_val_score(
        clf, TRN_X, TRN_Y, 
        cv=int(KFOLD), 
        scoring=metrics.make_scorer(metrics.r2_score)
    )

    (r2, rmse, mae) = (
        metrics.r2_score(VAL_Y, PRD_Y),
        metrics.mean_squared_error(VAL_Y, PRD_Y, squared=False),
        metrics.mean_absolute_error(VAL_Y, PRD_Y)
    )

    with open(path.join(argv[4], argv[1] + '_' + argv[2] + '_' + argv[3] + '.txt'), 'w') as f:
        with redirect_stdout(f):
            print('* Train/Validate entries: `{}/{} ({})`'.format(TRN_L, VAL_L, TRN_L+VAL_L))
            print('* Cross-validation R2: `%0.2f (+/-%0.2f)`'%(kScores.mean(), kScores.std()*2))
            print('* R2 score: `{:.2f}`'.format(r2))
            print('* Root Mean square error: `{:.2f}`'.format(rmse))
            print('* Mean absolute error: `{:.2f}`'.format(mae))