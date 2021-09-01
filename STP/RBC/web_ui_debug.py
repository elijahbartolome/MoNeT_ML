from scipy.sparse import data
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import STP_constants as cst
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, VotingRegressor, VotingClassifier,GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier, StackingClassifier, StackingRegressor, BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import requests 
from io import StringIO

st.title("STP Models")

st.session_state.dataset_name = st.sidebar.selectbox("Select Dataset", ("", "SCA", "CLS", "REG"))

st.session_state.model_name = st.sidebar.selectbox("Select Model", ("", "B", "ET", "GBT", "RF", "S", "V"))

st.sidebar.write("""
## Model Abbreviations:
* `b`: Bagging Classifier/Regressor
* `et`: Extra Trees
* `gbt`: Gradient Boosted Trees
* `rf`: Random Forest
* `s`: Stacking Classifier/Regressor
* `v`: Voting Classifier/Regressor
""")

st.session_state.metric_name = st.sidebar.selectbox("Select Metric", ("", "CPT", "TTI", "TTO", "WOP", "POE", "MIN", "RAP"))

st.sidebar.write("""
## Metric Abbreviations:
* `CPT`: Cumulative fraction of mosquitoes divided by time
* `TTI`: Time to introgression
* `TTO`: Time to outrogression
* `WOP`: Window of protection
* `POE`: Probability of elimination/fixation
* `MIN`: Minimum of mosquitoes
* `RAP`: Fraction of mosquites at timepoint
""")

(VT_SPLIT, TREES, DEPTH, KFOLD, JOB) = (
        cst.VT_SPLIT, cst.TREES, cst.DEPTH, cst.KFOLD, cst.JOB
    )
@st.cache
def get_dataset(dataset):

    ###############################################################################
    # Read CSV
    ###############################################################################
    if dataset == "REG":
        url = "https://drive.google.com/file/d/1NUUYmBIJmW9mRybg4jU8daFqUJbFdc_e/view?usp=sharing"
    elif dataset == "CLS": 
        url = "https://drive.google.com/file/d/1SqmbYkrDx8-0GJdrXrxPioyH1WeKo8FO/view?usp=sharing"
    elif dataset == "SCA":
        url = "https://drive.google.com/file/d/1w9Ika6d4V_tQ3HyghAGCefji6ajuzzaA/view?usp=sharing"

    file_id = url.split('/')[-2]
    dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
    url2 = requests.get(dwn_url).text
    csv_raw = StringIO(url2)
    df = pd.read_csv(csv_raw)
    
    return df

@st.cache
def get_training_data(MTR, DATA):
    # Features and labels ---------------------------------------------------------
    COLS = list(DATA.columns)
    (FEATS, LABLS) = (
        [i for i in COLS if i[0]=='i'],
        [i for i in COLS if i[0]!='i']
    )
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

    return TRN_X, TRN_Y

if st.session_state.metric_name and st.session_state.dataset_name and st.session_state.model_name: 
    #dataset = get_dataset(st.session_state.metric_name, st.session_state.dataset_name)

    dataset = get_dataset(st.session_state.dataset_name)
    st.write(dataset)
    TRN_X, TRN_Y = get_training_data(st.session_state.metric_name, dataset)

    def add_parameter_ui():
        params = []

        if st.session_state.dataset_name == "SCA":
            FEATS = ['i_sex', 'i_ren', 'i_res', 'i_rsg', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr', 'i_hrm', 'i_hrf', 'i_grp', 'i_mig']
        elif st.session_state.dataset_name == 'CLS' or st.session_state.dataset_name == 'REG':
            FEATS = ['i_sxm', 'i_sxg', 'i_sxn', 'i_ren', 'i_res', 'i_rsg', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr', 'i_hrm', 'i_hrf', 'i_grp', 'i_mig']

        st.sidebar.write("Please choose inputs for features to make prediction on:")

        for feat in FEATS:
            value = st.sidebar.text_input(feat)
            params.append(value)

        test = np.array(params)
        test = test.reshape(1, -1)
        return test

    params = add_parameter_ui()

    @st.cache
    def get_model(dataset, model):
        # regression model
        if dataset == "REG" or dataset == "SCA":
            if model == 'B':
                clf = BaggingRegressor(
                    n_estimators=TREES
                )   
            elif model == 'ET':
                clf = ExtraTreesRegressor(
                    n_estimators=TREES, max_depth=DEPTH,
                    min_samples_split=5, min_samples_leaf=50,
                    max_features=None, max_leaf_nodes=None,
                    n_jobs=JOB
                )
            elif model == 'GBT':
                clf = GradientBoostingRegressor(
                    n_estimators=TREES,
                    min_samples_split=5, min_samples_leaf=50,
                    max_features=None, max_leaf_nodes=None
                )
            elif model == 'RF':
                clf = RandomForestRegressor(
                    n_estimators=TREES, max_depth=DEPTH,
                    min_samples_split=5, min_samples_leaf=50,
                    max_features=None, max_leaf_nodes=None,
                    n_jobs=JOB
                )
            elif model == 'S':
                rf = RandomForestRegressor(
                n_estimators=TREES, max_depth=DEPTH,
                min_samples_split=5, min_samples_leaf=50,
                max_features=None, max_leaf_nodes=None,
                n_jobs=JOB
                )
                estimators = [
                    ('svr', LinearSVR(random_state=42)),
                    ('lr', RidgeCV())
                ]
                clf = StackingRegressor(
                    estimators = estimators, final_estimator = rf,
                    n_jobs = JOB
                )
            elif model == 'V':
                rf = RandomForestRegressor(
                    n_estimators=TREES, max_depth=DEPTH,
                    min_samples_split=5, min_samples_leaf=50,
                    max_features=None, max_leaf_nodes=None,
                    n_jobs=JOB
                )
                estimators = [('lr', LinearRegression()), ('rf', rf)]
                clf = VotingRegressor(estimators=estimators, n_jobs=JOB)
        # classifier model
        else:
            if model == 'B':
                clf = BaggingClassifier(
                    n_estimators=TREES
                )
            elif model == 'ET':
                clf = ExtraTreesClassifier(
                    n_estimators=TREES, max_depth=DEPTH, criterion='entropy',
                    min_samples_split=5, min_samples_leaf=50,
                    max_features=None, max_leaf_nodes=None,
                    n_jobs=JOB, bootstrap=True
                )
            elif model == 'GBT':
                clf = GradientBoostingClassifier(
                    n_estimators=TREES, max_depth=DEPTH,
                    min_samples_split=5, min_samples_leaf=50,
                    max_features=None, max_leaf_nodes=None
                )
            elif model == 'RF':
                clf = RandomForestClassifier(
                    n_estimators=TREES, max_depth=DEPTH, criterion='entropy',
                    min_samples_split=5, min_samples_leaf=50,
                    max_features=None, max_leaf_nodes=None,
                    n_jobs=JOB, bootstrap=True
                )
            elif model == 'S':
                rf = RandomForestClassifier(
                    n_estimators=TREES, max_depth=DEPTH, criterion='entropy',
                    min_samples_split=5, min_samples_leaf=50,
                    max_features=None, max_leaf_nodes=None,
                    n_jobs=JOB
                )
                estimators = [('rf', rf), ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))]
                clf = StackingClassifier(
                    estimators = estimators, n_jobs = JOB
                )
            elif model == 'V':
                rf = RandomForestClassifier(
                    n_estimators=TREES, max_depth=DEPTH, criterion='entropy',
                    min_samples_split=5, min_samples_leaf=50,
                    max_features=None, max_leaf_nodes=None,
                    n_jobs=JOB
                )
                v_estimators = [('rf', rf), 
                ('lr', LogisticRegression(multi_class='multinomial', random_state = 1)),
                ('gnb', GaussianNB())
                ]
                clf = VotingClassifier(estimators=v_estimators, voting='hard', n_jobs=JOB)

        return clf

    clf = get_model(st.session_state.dataset_name, st.session_state.model_name)

    def predict(clf, TRN_X, TRN_Y, params, metric):
        clf.fit(TRN_X, TRN_Y)
        output = clf.predict(params)
        st.write("""
        ## Predicted Output for
        """)
        st.write(metric)
        st.write(output[0])

    # Run functions
    run_clicked = st.sidebar.button('Run model')

    if run_clicked: 
        predict(clf, TRN_X, TRN_Y, params, st.session_state.metric_name)