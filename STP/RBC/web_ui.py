import streamlit as st
import joblib
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

st.title("STP Models")

st.session_state.dataset_name = st.sidebar.selectbox("Select Dataset", ("", "SCA", "CLS", "REG"))

st.session_state.model_name = st.sidebar.selectbox("Select Model", ("", "ET", "RF"))

st.sidebar.write("""
## Model Abbreviations (Only B, RF, and ET working now):
* `b`: Bagging Classifier/Regressor
* `et`: Extra Trees
* `gbt`: Gradient Boosted Trees
* `rf`: Random Forest
* `s`: Stacking Classifier/Regressor
* `v`: Voting Classifier/Regressor
""")

st.session_state.metric_name = st.sidebar.selectbox("Select Metric", ("", "CPT"))

st.sidebar.write("""
## Metric Abbreviations (Only CPT working now):
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

if st.session_state.metric_name and st.session_state.dataset_name and st.session_state.model_name: 

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
    def get_model(dataset, model, metric):
        file_name = "{dataset_name}_{model_name}_{metric_name}.pkl".format(dataset_name=dataset, model_name=model, metric_name=metric)
        clf = joblib.load(file_name)

        return clf

    clf = get_model(st.session_state.dataset_name, st.session_state.model_name, st.session_state.metric_name)

    def predict(clf, params, metric):
        output = clf.predict(params)
        st.write("""
        ## Predicted Output for
        """)
        st.write(metric)
        st.write(output[0])

    # Run functions
    run_clicked = st.sidebar.button('Run model')

    if run_clicked: 
        predict(clf, params, st.session_state.metric_name)