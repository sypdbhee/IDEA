# IDEA
The code for performing dereverberation

# Extracting log power spectrum: python build_data.py (Reverber_Inp) (LPS_Output)
python build_data.py ~/IDEA/train/3 ~/IDEA/Data/train/noisy_3
python build_data.py ~/IDEA/train/6 ~/IDEA/Data/train/noisy_6
python build_data.py ~/IDEA/train/9 ~/IDEA/Data/train/noisy_9
python build_data.py ~/IDEA/train/clean ~/IDEA/Data/train/clean

# Training process: 
# EP stage: 
python main.py --mode train --path 3/ --data 3 --gpus 0
python main.py --mode train --path 6/ --data 6 --gpus 0
python main.py --mode train --path 9/ --data 9 --gpus 0

# EI stage:
python main.py --mode s_train --path soft/ --gpus 0

# Testing process:
python main.py --mode s_test --path soft/ --data 3 --gpus 0
python main.py --mode s_test --path soft/ --data 6 --gpus 0
python main.py --mode s_test --path soft/ --data 9 --gpus 0
python main.py --mode s_test --path soft/ --data 4 --gpus 0
python main.py --mode s_test --path soft/ --data 7 --gpus 0
python main.py --mode s_test --path soft/ --data 10 --gpus 0

# Please cite the paper:
W. Lee, S. Wang, F. Chen, X. Lu, S. Chien and Y. Tsao, "Speech Dereverberation Based on Integrated Deep and Ensemble Learning Algorithm," in Proc. ICASSP, pp. 5454-5458, 2018
