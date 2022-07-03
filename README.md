## About the project
Despite the fact that the number of transaction fraud events grows currently slower
than number of transactions in total, it is still a problem for many institutions. 
Detecting fraudulent transactions is a particularly difficult task given multiple reasons
such as general lack of labels, class imbalance, hidden and evolving fraud patterns.
Even more difficulties emerge while modeling public transaction datasets, namely
feature anonymization, missing information, and aggregation of data. This work
suggests a pipeline of modeling fraudulent transactions which accounts for most of
those concerns based on other researchers’ experience. From the group of modeling
approaches, one can distinguish those based on transaction features and those using
graph anomaly detection methods. This research combines both of those methods and 
presents cross-validation results over two datasets. Performance scores themselves do not indicate 
outstanding predictive power of any presented approach. Nevertheless, addition of graph features in case of the 
second dataset significantly improved validation scores, and therefore indicated the direction for further research.

## Links

[[Vesta raw dataset]](https://www.kaggle.com/competitions/ieee-fraud-detection)

[[Elliptic raw dataset]](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

---

[[project directory structure]](https://drivendata.github.io/cookiecutter-data-science/)

[[miceforest imputation method]](https://morioh.com/p/e19cd87c66e3)

[[Explanation of HITS algorithm]](https://www.math.ucdavis.edu/~saito/courses/167.s17/Lecture24.pdf)

[[Great YouTube channel explaining centrality and community algorithms]](https://www.youtube.com/channel/UCHjLtIISxuvj2QDKTS_UYcg)

## Further research
- [ ] Optimize hyperparameter tuning using cuML API to train models
- [ ] Entity embedding method applied within cross validation function
- [ ] Evaluate Graph Neural Network (GNN) methods
