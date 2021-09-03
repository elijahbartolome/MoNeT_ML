#!/bin/bash
# chmod +x STP_clsTrain.sh

# Train models
python STP_Train_rf.py 'SCA' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'SCA' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'CLS' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'CLS' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'REG' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'REG' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'


python STP_Train_rf.py 'SCA' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'SCA' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'CLS' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'CLS' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'REG' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'REG' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'


python STP_Train_gbt.py 'SCA' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'SCA' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'SCA' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_gbt.py 'CLS' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'CLS' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'CLS' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_gbt.py 'REG' 'CPT' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'REG' 'TTI' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'REG' 'TTO' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'