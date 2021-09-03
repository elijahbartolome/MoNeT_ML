import streamlit as st
import joblib
import numpy as np
import STP_constants as cst
from pathlib import Path

st.title("STP Models")

st.session_state.dataset_name = st.sidebar.selectbox("Select Dataset", ("", "SCA", "CLS", "REG"))

st.session_state.model_name = st.sidebar.selectbox("Select Model", ("", "ET", "GBT", "RF"))

st.sidebar.write("""
## Model Abbreviations (Only RF, GBT, and ET working now):
* `b`: Bagging Classifier/Regressor
* `et`: Extra Trees
* `gbt`: Gradient Boosted Trees
* `rf`: Random Forest
* `s`: Stacking Classifier/Regressor
* `v`: Voting Classifier/Regressor
""")

st.session_state.metric_name = st.sidebar.selectbox("Select Metric", ("", "CPT", "TTI", "TTO"))

st.sidebar.write("""
## Metric Abbreviations (Only CPT, TTI, and TTO working now):
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

#get_clicked = st.sidebar.button('Get model')

if st.session_state.metric_name and st.session_state.dataset_name and st.session_state.model_name: 

    @st.cache
    def get_model(dataset, model, metric):
        file_name = Path(__file__).parents[0] / "input/{dataset_name}_{model_name}_{metric_name}.pkl".format(dataset_name=dataset, model_name=model, metric_name=metric)
        clf = joblib.load(file_name)

        return clf

    clf = get_model(st.session_state.dataset_name, st.session_state.model_name, st.session_state.metric_name)

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