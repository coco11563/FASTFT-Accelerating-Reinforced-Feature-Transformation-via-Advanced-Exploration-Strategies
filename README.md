# FastFT: Accelerating Reinforced Feature Transformation via Advanced Exploration Strategies
## Basic info:
This is the release code for :
[FastFT: Accelerating Reinforced Feature Transformation via Advanced Exploration Strategies](http://arxiv.org/abs/2503.20394)
which is accepted by ICDE 2025!


Recommended ref:
```
Tianqi He, Xiaohan Huang, Yi Du, Qingqing Long, Ziyue Qiao, Min Wu, Yanjie Fu, Yuanchun Zhou, Meng Xiao. FastFT: Accelerating Reinforced Feature Transformation via Advanced Exploration Strategies. IEEE 41th International Conference on Data Engineering (ICDE), 2025
```

Recommended Bib:
```
ICDE version bib:
@inproceedings{he2025fastft,
  title={FastFT: Accelerating Reinforced Feature Transformation via Advanced Exploration Strategies},
  author={He, Tianqi and Huang, Xiaohan and Du, Yi  and Long, Qingqing and Ziyue, Qiao and Min, Wu and Yanjie, Fu and Yuanchun, Zhou and Meng, Xiao},
  booktitle={2024 IEEE 40th International Conference on Data Engineering (ICDE)},
  pages={1--14},
  year={2025},
  organization={IEEE}
}
```
***
## Paper Abstract

Feature Transformation is crucial for classic machine learning that aims to generate feature combinations to enhance the performance of downstream tasks from a data-centric perspective. Current methodologies, such as manual expert-driven processes, iterative-feedback techniques, and exploration-generative tactics, have shown promise in automating the feature transformation workflow by minimizing human involvement.
However, three challenges remain in those frameworks: (1) It predominantly depends on downstream task performance metrics, as assessment is time-consuming, especially for large datasets. (2) The diversity of feature combinations will hardly be guaranteed after random exploration ends. (3) Rare significant transformations lead to sparse valuable feedback that hinders the learning processes or leads to less effective results. 
In response to these challenges, we introduce FastFT, an innovative framework that leverages a trio of advanced strategies. We first decouple the feature transformation evaluation from the outcomes of the generated datasets via the performance predictor. 
To address the issue of reward sparsity, we developed a method to evaluate the novelty of generated transformation sequences. Incorporating this novelty into the reward function accelerates the model's exploration of effective transformations, thereby improving the search productivity. 
Additionally, we combine novelty and performance to create a prioritized memory buffer, ensuring that essential experiences are effectively revisited during exploration. Our extensive experimental evaluations validate the performance, efficiency, and traceability of our proposed framework, showcasing its superiority in handling complex feature transformation tasks.
***

## FastFT Performance
The best results are highlighted in **bold**. The second-best results are highlighted in <u>underline</u>. 

We reported F1-Score for classification tasks (C), 1-RAE for regression tasks (R), and AUC for detection tasks (D). The
t-statistics and p-values comparing the performances of each baseline with FASTFT are presented in the last two rows.

 \* The standard deviation is computed based on the results of 5 independent runs.

† Methods marked by "×" indicate that their execution time is unacceptably prolonged on some of the selected datasets.
| Name | Source | Task | Samples | Features | RFG | ERG | LDA | AFT† | NFS | TTG | DIFER† | OpenFE | CAAFE | GRFG† | FastFT* |
|------|--------|------|---------|----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Alzheimers | Kaggle | C | 2149 | 33 | 0.936 | <u>0.956</u> | 0.584 | 0.907 | 0.914 | 0.925 | 0.952 | 0.951 | 0.945 | 0.953 | **0.974**±**0.010** |
| Cardiovascular | Kaggle | C | 5000 | 12 | <u>0.720</u> | 0.709 | 0.561 | 0.712 | 0.710 | 0.709 | 0.712 | 0.706 | 0.711 | **0.722** | **0.722**±**0.015** |
| Fetal Health | Kaggle | C | 2126 | 22 | 0.913 | 0.917 | 0.744 | 0.918 | 0.914 | 0.709 | 0.944 | 0.943 | 0.945 | <u>0.951</u> | **0.954**±**0.008** |
| Pima Indian | UCIrvine | C | 768 | 8 | 0.693 | 0.703 | 0.676 | 0.736 | 0.762 | 0.747 | 0.746 | 0.744 | 0.755 | <u>0.776</u> | **0.789**±**0.035** |
| SVMGuide3 | LibSVM | C | 1243 | 21 | 0.703 | 0.747 | 0.683 | 0.829 | 0.831 | 0.766 | 0.773 | 0.831 | 0.828 | <u>0.850</u> | **0.863**±**0.028** |
| Amazon Employee | Kaggle | C | 32769 | 9 | 0.744 | 0.740 | 0.920 | 0.943 | 0.935 | 0.806 | 0.937 | 0.944 | 0.943 | <u>0.946</u> | **0.951**±**0.002** |
| German Credit | UCIrvine | C | 1001 | 24 | 0.695 | 0.661 | 0.627 | 0.751 | <u>0.765</u> | 0.731 | 0.752 | 0.757 | 0.759 | 0.763 | **0.777**±**0.015** |
| Wine Quality Red | UCIrvine | C | 999 | 12 | 0.599 | 0.611 | 0.600 | 0.658 | 0.666 | 0.647 | 0.675 | 0.658 | 0.677 | <u>0.686</u> | **0.692**±**0.031** |
| Wine Quality White | UCIrvine | C | 4898 | 12 | 0.552 | 0.587 | 0.571 | 0.673 | 0.679 | 0.638 | 0.681 | 0.670 | 0.676 | <u>0.685</u> | **0.691**±**0.005** |
| Jannis | AutoML | C | 83733 | 55 | <u>0.714</u> | 0.712 | 0.477 | 0.695 | <u>0.714</u> | 0.711 | × | 0.708 | 0.698 | × | **0.722**±**0.003** |
| OpenML\_618 | OpenML | R | 1000 | 50 | 0.415 | 0.427 | 0.372 | 0.665 | 0.640 | 0.587 | 0.644 | 0.717 | <u>0.725</u> | 0.672 | **0.786**±**0.015** |
| OpenML\_589 | OpenML | R | 1000 | 25 | 0.638 | 0.560 | 0.331 | 0.672 | 0.711 | 0.682 | 0.715 | 0.719 | 0.714 | <u>0.753</u> | **0.768**±**0.017** |
| OpenML\_616 | OpenML | R | 500 | 50 | 0.448 | 0.372 | 0.385 | 0.585 | 0.593 | 0.559 | 0.556 | 0.632 | <u>0.647</u> | 0.603 | **0.726**±**0.031** |
| OpenML\_607 | OpenML | R | 1000 | 50 | 0.579 | 0.406 | 0.376 | 0.658 | 0.675 | 0.639 | 0.636 | <u>0.730</u> | 0.651 | 0.680 | **0.764**±**0.043** |
| WBC | UCIrvine | D | 278 | 30 | 0.753 | 0.766 | 0.736 | 0.743 | 0.755 | 0.752 | <u>0.956</u> | 0.905 | 0.601 | 0.785 | **0.972**±**0.058** |
| Mammography | OpenML | D | 11183 | 6 | 0.731 | 0.728 | 0.668 | 0.714 | 0.728 | 0.734 | 0.532 | <u>0.806</u> | 0.668 | 0.751 | **0.860**±**0.036** |
| Thyroid | UCIrvine | D | 3772 | 6 | 0.813 | 0.790 | 0.778 | 0.797 | 0.722 | 0.720 | 0.613 | <u>0.967</u> | 0.776 | 0.954 | **0.999**±**0.008** |
| SMTP | UCIrvine | D | 95156 | 3 | 0.885 | 0.836 | 0.765 | 0.881 | 0.816 | 0.895 | 0.573 | 0.494 | 0.732 | <u>0.943</u> | **0.950**±**0.061** |




## How to run:

### step 0: install requirements:
We recommend Python version 3.10 with torch version 1.13.1. For reproducing FastFT, please install the tools with exactly version.
```
pip install -r requirement.txt
```

### step 1: download the code and dataset:
```
git clone https://github.com/coco11563/FASTFT-Accelerating-Reinforced-Feature-Transformation-via-Advanced-Exploration-Strategies.git
```
then:
```
follow the instruction in readme.md in `/data/processed/data_info.md` to get the dataset
```

### step 2: run the code with main script:`main.py`

```
python3 main_robust.py --name DATASETNAME --episodes SEARCH_EP_NUM --steps SEARCH_STEP_NUM...
```

please check each configuration in `initial.py`

### step 3: enjoy the generated dataset:

the generated feature will in ./tmp/NAME_TRAIN_MODE/best.csv
