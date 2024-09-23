# Semi-Supervised Dual-Stream Self-Attentive Adversarial Graph Contrastive Learning for Cross-Subject EEG-based Emotion Recognition 
*   A Pytorch implementation of our under reviewed paper "Semi-Supervised Dual-Stream Self-Attentive Adversarial Graph Contrastive Learning for Cross-Subject EEG-based Emotion Recognition".
*   [arxiv](https://arxiv.org/abs/2308.11635 "")
# Installation
*   Python 3.7
*   Pytorch 1.3.1
*   NVIDIA CUDA 9.2
*   Numpy 1.20.3
*   Scikit-learn 0.23.2
*   scipy 1.3.1
# Databases
*   [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html ""), [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html ""), [SEED-V](https://bcmi.sjtu.edu.cn/~seed/seed-v.html "") and [FACED](https://doi.org/10.7303/syn50614194 "")
# Training
*   DS-AGC model definition file: models_DS_AGC.py
*   Pipeline of the DS-AGC: implementation_DS_AGC.py
*   Implementation of domain adversarial training: adversarial.py
*   Self-attention model: self_attention.py
# Usage
*   After modify setting (path, etc), just run the main function in the implementation_DS-AGC.py
# Citation
* @misc{ye2023semi,
  title={Semi-Supervised Dual-Stream Self-Attentive Adversarial Graph Contrastive Learning for Cross-Subject EEG-based Emotion Recognition},
  author={Ye, Weishan and Zhang, Zhiguo and Zhang, Min and Teng, Fei and Zhang, Li and Li, Linling and Huang, Gan and Wang, Jianhong and Ni, Dong and Liang, Zhen},
  year={2023},
  eprint={2308.11635},
  archivePrefix={arXiv}
}
