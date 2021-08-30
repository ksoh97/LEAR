# Learn-Explain-Reinforce
Tensorflow implementation of [Learn-Explain-Reinforce: Counterfactual Reasoning and Its Guidance to Reinforce an Alzheimer's Disease Diagnosis Model](https://arxiv.org/abs/2108.09451).


### Requirements
tensorflow (2.2.0)\
tensorboard (2.2.2)\
tensorflow-addons (0.11.0)\
tqdm (4.48.0)\
matplotlib (3.3.0)\
numpy (1.19.0)\
scikit-learn (0.23.2)


### Datasets
Place them into "data_path" on each Config.py
1. [HandWritten digits data (MNIST)](http://yann.lecun.com/exdb/mnist/)
2. [Alzheimer’s Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/)


### How to run
Mode:\
#0 Learn / #1 Explain / #2 Reinforce / #3 Iter_Explanation / #4 Iter_Reinforcement

1. Learn: pre-training a diagnostic model
>- Diagnosic models are a CNN-based architecure (ResNet18, VoxCNN, and SonoNet16)
  >- `training.py --mode=0`

2. Explain: Counterfactual map generation using a pre-trained diagnostic model
>- Set the classifier and encoder weight for training (freeze)
>- Change the mode from 0 to 1 on Config.py
  >- `training.py --mode=1`

3. Reinforce: Explanation-guided attention to improve its generalizabiliy and performance
>- Set the counterfactual map generator (CMG) weight for training an explanation-guided attention (XGA) module injected into a diagnostic model
>- Change the mode to 2 on Config.py
  >- `training.py --mode=2`

4. Iterative explanation-reinforcement learning
>- Enhances the quality of visual explanation as well as the performance of the diagnostic model
>- Change the mode to 3 or 4 on Config.py
  >- `training.py --mode=3 and --mode=4`


### Config.py of each dataset with saved weight path
data_path = Raw dataset path\
save_path = Storage path to save results such as tensorboard event files, model weights, etc.\
cls_weight_path = Pre-trained diagnostic model weight path trained in mode#0 setup\
enc_weight_path = Pre-trained encoder weight path trained in mode#0 setup\
cmg_weight_path = Pre-trained counterfactual map generator weight path trained in mode#1 setup\
xga_cls_weight_path = Pre-trained XGA-injected diagnostic model weight path trained in mode#2 setup\
xga_enc_weight_path = Pre-trained XGA-injected encoder weight path trained in mode#2 setup
