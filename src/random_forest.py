from src.setup_logger import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, cross_validate
from imblearn.ensemble import BalancedRandomForestClassifier # noqa
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, \
    roc_curve, auc, precision_recall_curve
from lightgbm import LGBMClassifier # noqa
import pickle
import matplotlib.pyplot as plt
import re
import os
logger = logging.getLogger("RandomForest")


class RF:
    def __init__(self, imp_datasets_dir="../data/processed/", y_train_path="../data/processed/ieee_train_y.pkl",
                 target="isFraud", n_jobs=-1):
        self.imp_path = imp_datasets_dir
        self.y_train_path = y_train_path
        self.y_train = None
        self.target = target
        self.X_train_list, self.X_val_list, self.y_train_list, self.y_val_list = [], [], [], []
        self.cv_n_splits, self.cv = None, None
        self.scoring = {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score),
            "recall": make_scorer(recall_score),
            "f1_score": make_scorer(f1_score)
        }
        self.cv_base_models, self.cv_scores, self.tuned_models = [], [], []
        self.n_jobs = n_jobs
        self.cv_res = []

    def load_data(self, dataset_final_pattern=r"ieee_train_final_", with_pickle=True):
        # if exclude_cols is None:
        #     exclude_cols = []
        # Search for imputed datasets
        path_list = os.listdir(self.imp_path)

        if with_pickle:
            x_path_list = list(filter(re.compile(dataset_final_pattern+r"\.pkl").match, path_list))

            # Append those datasets to list
            for p in x_path_list:
                with open(self.imp_path+p, "rb") as h:
                    self.X_train_list.append(pickle.load(h))
            # Load target vector
            with open(self.y_train_path, "rb") as h:
                self.y_train = pickle.load(h)

        # Otherwise, load from .csv
        else:
            x_path_list = list(filter(re.compile(dataset_final_pattern+r"\d"+r"\.csv").match, path_list))
            for p in x_path_list:
                self.X_train_list.append(pd.read_csv(self.imp_path+p))

            self.y_train = pd.read_csv(self.y_train_path)

        return 0

    def cv_base_model(self, verbose=1, save_model=True, name_prefix="vesta_baseline_rf_", print_val_scoring=False,
                      return_scoring=False, seed=2022, cv_n_splits=10):
        for i, X in enumerate(self.X_train_list):
            self.cv = RepeatedStratifiedKFold(n_splits=cv_n_splits, random_state=seed, n_repeats=3)
            model = RandomForestClassifier(n_jobs=self.n_jobs, class_weight="balanced_subsample", random_state=seed)
            # model = BalancedRandomForestClassifier(n_jobs=self.n_jobs, random_state=self.seed)
            # model = LGBMClassifier(
            #     n_estimators=400, class_weight="balanced", n_jobs=self.n_jobs, random_state=self.seed
            # )

            cv_scores = cross_validate(
                model, X, self.y_train, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs, error_score='raise',
                verbose=verbose, return_estimator=True
            )
            cv_scores_copy = cv_scores.copy()
            for k in cv_scores.keys():
                if k in ["fit_time", "score_time", "estimator"]:
                    cv_scores_copy.pop(k)

            if print_val_scoring:
                print(f"""
                    Mean validation results:
                    Accuracy: {np.round(np.mean(cv_scores["test_accuracy"]), 4)}
                    Precision: {np.round(np.mean(cv_scores["test_precision"]), 4)}
                    Recall: {np.round(np.mean(cv_scores["test_recall"]), 4)}
                    F1 score: {np.round(np.mean(cv_scores["test_f1_score"]), 4)}
                """)

            self.cv_scores.append(cv_scores)
            self.cv_base_models.append(model)

            if save_model:
                if not os.path.exists("../models/"):
                    os.mkdir("../models/")
                with open(f"../models/{name_prefix}{i}.pkl", "wb") as handle:
                    pickle.dump(self.cv_base_models[i], handle, protocol=pickle.HIGHEST_PROTOCOL)

        if return_scoring:
            return cv_scores_copy # noqa

        return 0

    def tune_model(self, verbose=1, print_val_scoring=True, n_param_samples=20, save_model=True,
                   name_prefix="ieee_baseline_rf_tuned_", seed=2022):

        for i in range(len(self.cv_base_models)):

            n_estimators = [50, 100, 200, 250, 300, 400, 500, 800]
            max_features = ['auto', 'sqrt']
            max_depth = [5, 10, 20, 30, 40, 50, 70, 90]
            min_samples_split = [2, 4, 5, 10, 12]
            min_samples_leaf = [1, 2, 4, 7, 9]
            bootstrap = [True, False]

            random_grid = {
                "n_estimators": n_estimators,
                "max_features": max_features,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "bootstrap": bootstrap
            }

            model_tuned = RandomizedSearchCV(
                estimator=self.cv_base_models[i], param_distributions=random_grid, n_iter=n_param_samples,
                scoring=self.scoring, cv=self.cv, verbose=verbose, random_state=seed,
                n_jobs=self.n_jobs, error_score="raise", refit="f1_score"
            )

            model_tuned.fit(self.X_train_list[i], self.y_train)

            self.tuned_models.append(model_tuned)

            self.cv_res.append(self.tuned_models[0].cv_results_)

            if print_val_scoring:
                res_dict = self.tuned_models[0].cv_results_
                print(f"""
                Mean validation results:
                Accuracy: {np.round(res_dict["mean_test_accuracy"][0], 4)}
                Precision: {np.round(res_dict["mean_test_precision"][0], 4)}
                Recall: {np.round(res_dict["mean_test_recall"][0], 4)}
                F1 score: {np.round(res_dict["mean_test_f1_score"][0], 4)}
                """)

            if save_model:
                if not os.path.exists("../models/"):
                    os.mkdir("../models/")
                with open(f"../models/{name_prefix}{i}.pkl", "wb") as handle:
                    pickle.dump(self.tuned_models[i], handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0

    def get_probas(self, pick_first_iter=True, tuned_model=None, x_test_set=None):
        if pick_first_iter:
            return self.tuned_models[0].best_estimator_.predict_proba(self.X_train_list[0])
        else:
            return tuned_model.best_estimator_.predict_proba(x_test_set)

    @staticmethod
    def plot_roc(pred_probas, real_labels, save_plot=False, save_dir=r"visualizations/", save_name="roc_auc_plot"):
        fpr, tpr, threshold = roc_curve(real_labels, pred_probas)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        if save_plot:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            plt.savefig(save_dir+save_name)

        plt.show()

        return 0

    @staticmethod
    def plot_precision_recall(pred_probas, real_labels,
                              save_plot=False, save_dir=r"visualizations/", save_name="roc_auc_plot"):

        precision, recall, thresholds = precision_recall_curve(real_labels, pred_probas)

        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='purple')

        ax.set_title('Precision-Recall Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')

        if save_plot:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            plt.savefig(save_dir+save_name)

        plt.show()

        return 0
