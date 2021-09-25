#!/bin/bash
# chmod +x STP_clsTrain.sh

# Train models
<<comment
python STP_Train_rf.py 'SCA' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'SCA' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'SCA' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'CLS' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'CLS' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'CLS' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'REG' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'REG' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'REG' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'


python STP_Train_rf.py 'SCA' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'SCA' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'SCA' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'CLS' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'CLS' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'CLS' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'REG' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'REG' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'REG' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'


python STP_Train_rf.py 'SCA' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'SCA' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'SCA' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'CLS' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'CLS' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'CLS' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'REG' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'REG' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'REG' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'


python STP_Train_rf.py 'SCA' 'WOP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'SCA' 'WOP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'SCA' 'WOP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'CLS' 'WOP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'CLS' 'WOP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'CLS' 'WOP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'REG' 'WOP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'REG' 'WOP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'REG' 'WOP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'


python STP_Train_rf.py 'SCA' 'POE' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'SCA' 'POE' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'SCA' 'POE' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'CLS' 'POE' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'CLS' 'POE' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'CLS' 'POE' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'REG' 'POE' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'REG' 'POE' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'REG' 'POE' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
comment


python STP_evaluate.py 'CLS' 'RF' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_evaluate.py 'CLS' 'ET' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_evaluate.py 'CLS' 'GBT' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_evaluate.py 'CLS' 'RF' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_evaluate.py 'CLS' 'ET' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_evaluate.py 'CLS' 'GBT' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_evaluate.py 'CLS' 'RF' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_evaluate.py 'CLS' 'ET' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_evaluate.py 'CLS' 'GBT' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_evaluate.py 'CLS' 'RF' 'WOP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_evaluate.py 'CLS' 'ET' 'WOP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_evaluate.py 'CLS' 'GBT' 'WOP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_evaluate.py 'CLS' 'RF' 'POE' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_evaluate.py 'CLS' 'ET' 'POE' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_evaluate.py 'CLS' 'GBT' 'POE' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'