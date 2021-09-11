#!/bin/bash
# chmod +x STP_clsTrain.sh

# Train models
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


python STP_Train_rf.py 'SCA' 'MIN' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'SCA' 'MIN' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'SCA' 'MIN' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'CLS' 'MIN' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'CLS' 'MIN' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'CLS' 'MIN' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'REG' 'MIN' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'REG' 'MIN' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'REG' 'MIN' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'


python STP_Train_rf.py 'SCA' 'RAP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'SCA' 'RAP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'SCA' 'RAP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'CLS' 'RAP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'CLS' 'RAP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'CLS' 'RAP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'

python STP_Train_rf.py 'REG' 'RAP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_et.py 'REG' 'RAP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
python STP_Train_gbt.py 'REG' 'RAP' 'C:\Users\azneb\MoNeT_ML\STP\RBC\input'
