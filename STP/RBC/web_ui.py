import streamlit as st
import joblib
import numpy as np
import STP_constants as cst
from pathlib import Path

st.title("STP Models")

st.write("""
## By Elijah Bartolome
""")

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

st.session_state.metric_name = st.sidebar.selectbox("Select Metric", ("", "CPT", "TTI", "TTO", "WOP", "POE"))

st.sidebar.write("""
## Metric Abbreviations:
* `CPT`: Cumulative fraction of mosquitoes divided by time
* `TTI`: Time to introgression
* `TTO`: Time to outrogression
* `WOP`: Window of protection
* `POE`: Probability of elimination/fixation
""")

(VT_SPLIT, TREES, DEPTH, KFOLD, JOB) = (
        cst.VT_SPLIT, cst.TREES, cst.DEPTH, cst.KFOLD, cst.JOB
    )

#get_clicked = st.sidebar.button('Get model')

if st.session_state.metric_name and st.session_state.dataset_name and st.session_state.model_name: 

    file_metric_name = Path(__file__).parents[0] / "input/{dataset_name}_{model_name}_{metric_name}.txt".format(dataset_name=st.session_state.dataset_name, model_name=st.session_state.model_name, metric_name=st.session_state.metric_name)
    if file_metric_name.is_file():
        metric_clicked = st.checkbox('See Model Metrics')
        if metric_clicked:
            file = open(file_metric_name)
            line = file.read()
            st.write("""
                ## Model Metrics:
                """)
            st.markdown(line)
            file.close()

    @st.cache(allow_output_mutation=True)
    def get_model(dataset, model, metric):
        file_name = Path(__file__).parents[0] / "input/{dataset_name}_{model_name}_{metric_name}.pkl".format(dataset_name=dataset, model_name=model, metric_name=metric)
        clf = joblib.load(file_name)

        return clf

    clf = get_model(st.session_state.dataset_name, st.session_state.model_name, st.session_state.metric_name)

    def add_parameter_ui():
        params = []

        i_sex = st.sidebar.slider(label="Sex-sorting of the released transgenic mosquitos" ,min_value=1, max_value=3, value=1, step=1)
        params.append(i_sex)

        i_ren = st.sidebar.slider(label="Number of releases (weekly)", min_value=0, max_value=24, value=12, step=1)
        params.append(i_ren)

        i_res = st.sidebar.slider(label="Release size (fraction of the total population)", min_value=0.0, max_value=1.0, value=.5, step=.01)
        params.append(i_res)

        i_rsg = st.sidebar.slider(label="Resistance generation rate", min_value=0.0, max_value=.1185, value=.079, step=.01)
        params.append(i_rsg)

        i_gsv = st.sidebar.slider(label="Genetic standing variation (value will be divided by 100)", min_value=0.0, max_value=1.0, value=1.0, step=.01)
        params.append(i_gsv/100)

        i_fch = st.sidebar.slider(label="Fitness cost on the H alleles (homing)", min_value=0.0, max_value=1.0, value=.175, step=.01)
        params.append(i_fch)

        i_fcb = st.sidebar.slider(label="Fitness cost on the B alleles (out-of-frame resistant)", min_value=0.0, max_value=1.0, value=.117, step=.01)
        params.append(i_fcb)

        i_fcr = st.sidebar.slider(label="Fitness cost on the R alleles (in-frame resistant)", min_value=0.0, max_value=1.0, value=0.0, step=.01)
        params.append(i_fcr)

        i_hrm = st.sidebar.slider(label="Homing rate on males", min_value=0.0, max_value=1.0, value=1.0, step=.01)
        params.append(i_hrm)

        i_hrf = st.sidebar.slider(label="Homing rate on females", min_value=0.0, max_value=1.0, value=.956, step=.01)
        params.append(i_hrf)

        i_grp = 0
        params.append(i_grp)

        i_mig = 0
        params.append(i_mig)

        return params

    st.sidebar.write("""
    ## Select Parameters:
    """)

    params = add_parameter_ui()

    def predict(clf, params, dataset, model, metric):
        if dataset == "REG" or dataset == "CLS":
            if params[0] == 1:
                new_params = [1, 0, 0]
                new_params.extend(params[1:])
            elif params[0] == 2:
                new_params = [0, 1, 0]
                new_params.extend(params[1:])
            elif params[0] == 3:
                new_params = [0, 0, 1]
                new_params.extend(params[1:])
        else:
            new_params = params

        new_params = np.array(new_params)
        new_params = new_params.reshape(1, -1)

        output = clf.predict(new_params)
        st.write("""
        ## Predicted Output for
        """)
        st.write("Dataset:")
        st.write(dataset)

        st.write("Model:")
        st.write(model)

        st.write("Metric:")
        st.write(metric)

        st.write("Output:")
        st.write(output[0])

    # Run functions
    run_clicked = st.sidebar.button('Run model')

    if run_clicked: 
        predict(clf, params, st.session_state.dataset_name, st.session_state.model_name, st.session_state.metric_name)