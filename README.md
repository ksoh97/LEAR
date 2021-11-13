# Learn-Explain-Reinforce
Tensorflow implementation of [Learn-Explain-Reinforce: Counterfactual Reasoning and Its Guidance to Reinforce an Alzheimer's Disease Diagnosis Model](https://arxiv.org/abs/2108.09451).

## Overall framework
- asdf
![Group 2835](https://user-images.githubusercontent.com/57162425/141603342-73bee29c-fa07-4c83-a919-bc69975a8414.jpg)

![Group 2850](https://user-images.githubusercontent.com/57162425/141603344-47c9900e-420b-46f9-9eee-7a4095d89041.jpg)

## Results
![Group 2777](https://user-images.githubusercontent.com/57162425/141603345-abdf11e0-f7bf-4ecf-979e-f1604cd27c2c.jpg)

![Group 2584](https://user-images.githubusercontent.com/57162425/141603337-4951d4d6-8237-4fc1-80dd-8c87f7dd9d18.png)

![Group 2896](https://user-images.githubusercontent.com/57162425/141603346-ec6afc03-9aa7-4f73-815a-d79969fd0f09.png)

## Requirements
tensorflow (2.2.0)\
tensorboard (2.2.2)\
tensorflow-addons (0.11.0)\
tqdm (4.48.0)\
matplotlib (3.3.0)\
numpy (1.19.0)\
scikit-learn (0.23.2)\
nibabel (3.0.1)


## Datasets
Place them into "data_path" on each Config.py
1. [HandWritten digits data (MNIST)](http://yann.lecun.com/exdb/mnist/)
2. [Alzheimer’s Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/)


## How to run
Mode:\
#0 Learn / #1 Explain / #2 Reinforce / #3 Iter_Explanation / #4 Iter_Reinforcement

1. Learn: pre-training a diagnostic model
>- Diagnosic models are a CNN-based architecure (ResNet18, VoxCNN, and SonoNet16)
  >- `LEAR_training.py --mode=0`

2. Explain: Counterfactual map generation using a pre-trained diagnostic model
>- Set the classifier and encoder weight for training (freeze)
>- Change the mode from 0 to 1 on Config.py
  >- `LEAR_training.py --mode=1`

3. Reinforce: Explanation-guided attention to improve its generalizabiliy and performance
>- Set the counterfactual map generator (CMG) weight for training an explanation-guided attention (XGA) module injected into a diagnostic model
>- Change the mode to 2 on Config.py
  >- `LEAR_training.py --mode=2`

4. Iterative explanation-reinforcement learning
>- Enhances the quality of visual explanation as well as the performance of the diagnostic model
>- Change the mode to 3 or 4 on Config.py
  >- `LEAR_iterative_training.py --mode=3 and --mode=4`


## Config.py of each dataset with saved weight path
data_path = Raw dataset path\
save_path = Storage path to save results such as tensorboard event files, model weights, etc.\
cls_weight_path = Pre-trained diagnostic model weight path trained in mode#0 setup\
enc_weight_path = Pre-trained encoder weight path trained in mode#0 setup\
gen_weight_path = Pre-trained counterfactual map generator weight path trained in mode#1 setup\
xga_cls_weight_path = Pre-trained XGA-injected diagnostic model weight path trained in mode#2 or mode#4 setup\
xga_enc_weight_path = Pre-trained XGA-injected encoder weight path trained in mode#2 or mode#4 setup\
xga_gen_weight_path = Pre-trained XGA-injected counterfactual map generator weight path trained in mode#3 setup
