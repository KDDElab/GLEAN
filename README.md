![IF_628](https://github.com/user-attachments/assets/64c70926-5767-4595-8a6d-aa1649978ffc)
# Generating Plausible Counterfactual Instances for Anomaly Interpretation with Generative Adversarial Networks

Note that this task is also referred to as outlier Interpretation, outlier aspect mining/discovering, outlier property detection, and outlier description.
We make the source code of the 11 baseline methods publicly available. In addition, since outlier-interpretation-main/GLEAN is the core code of the GLEAN model, if this paper is fortunate enough to be accepted, we promise to make it publicly available immediately upon acceptance. Thank you for your understanding.


### Twelve Outlier Explanation Methods

**This repository contains seven outlier interpretation methods: GLEAN[1], ATON [2], COIN[3], SiNNE[4], SHAP[5], LIME[6], Anchors[7], REVISE[8], DiCE[9], FACE[10], EACE[11], and PML[12].**

[1] Generating Plausible Counterfactual Instances for Anomaly Interpretation with Generative Adversarial Networks.

[2] Beyond Outlier Detection: Outlier Interpretation by Attention-Guided Triplet Deviation Network. In WWW. 2021.

[3] Contextual outlier interpretation. In IJCAI. 2018.

[4] A New Dimensionality-Unbiased Score For Efficient and Effective Outlying Aspect Mining. In Data Science and Engineering. 2022.

[5] A unified approach to interpreting model predictions. In NeuraIPS. 2017

[6] "Why should I trust you?" Explaining the predictions of any classifier. In SIGKDD. 2016.

[7] Anchors: High Precision Model-Agnostic Explanations. In AAAI. 2018.

[8] Towards realistic individual recourse and actionable explanations in black-box decision making systems. arXiv preprint arXiv:1907.09615. 2019.

[9] Explaining machine learning classifiers through diverse counterfactual explanations. In FAT. 2020.

[10] Face: feasible and actionable counterfactual explanations. In AIES. 2020.

[11]  Eace: explain anomaly via counterfactual explanations. In Pattern Recognition. 2025.

[12] Perspective-based multi-task learning for outlier interpretation.  In DASFAA. 2025.


### Requirements
main packages of this project  
```
torch==1.3.0
numpy==1.15.0
pandas==0.25.2
scikit-learn==0.23.1
pyod==0.8.2
tqdm==4.48.2
prettytable==0.7.2
shap==0.35.0
lime==0.2.0.1
alibi==0.5.5
```
