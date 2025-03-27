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


## How to run:
### step 1: download the code and dataset:
```
git clone git@github.com:coco11563/FASTFT-Accelerating-Reinforced-Feature-Transformation-via-Advanced-Exploration-Strategies.git
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
