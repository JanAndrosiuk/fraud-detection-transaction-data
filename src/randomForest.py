import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, RandomizedSearchCV, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve
import pickle
import matplotlib.pyplot as plt
import re
import os


class RF:
    def __init__(self, imp_datasets_dir="../data/processed/", y_train_path="../data/processed/ieee_train_y.pkl",
                 target="isFraud", seed=2022, cv_n_splits=10, n_jobs=-1):
        self.imp_path = imp_datasets_dir
        self.y_train_path = y_train_path
        self.y_train = None
        self.target = target
        self.X_train_list, self.X_val_list, self.y_train_list, self.y_val_list = [], [], [], []
        self.seed = seed
        self.cv_n_splits = cv_n_splits
        self.cv = RepeatedStratifiedKFold(n_splits=cv_n_splits, random_state=self.seed, n_repeats=3)
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
        # Search for imputed datasets
        path_list = os.listdir(self.imp_path)

        if with_pickle:
            X_path_list = list(filter(re.compile(dataset_final_pattern+r"\d"+r"\.pkl").match, path_list))

            # Append those datasets to list
            for p in X_path_list:
                with open(self.imp_path+p, "rb") as h:
                    self.X_train_list.append(pickle.load(h))

            # Load target vector
            with open(self.y_train_path, "rb") as h:
                self.y_train = pickle.load(h)

        # Otherwise, load from .csv
        else:
            X_path_list = list(filter(re.compile(dataset_final_pattern+r"\d"+r"\.csv").match, path_list))
            for p in X_path_list:
                self.X_train_list.append(pd.read_csv(self.imp_path+p))

            self.y_train = pd.read_csv(self.y_train_path)

        return 0

    def cv_base_model(self, verbose=1):
        for X in self.X_train_list:
            model = RandomForestClassifier()

            cv_scores = cross_validate(
                model, X, self.y_train, scoring=self.scoring, cv=self.cv,
                n_jobs=self.n_jobs, error_score='raise', verbose=verbose
            )

            self.cv_scores.append(cv_scores)
            self.cv_base_models.append(model)

        return 0

    def tune_model(self, verbose=1, print_val_scoring=True, n_param_samples=20):
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
                scoring=self.scoring, cv=self.cv, verbose=verbose, random_state=self.seed,
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

        return 0

    def save_tuned_models(self):
        if not os.path.exists("../models/"):
            os.mkdir("../models/")
        with open("../data/processed/ieee_baseline_tuned_rf.pkl", "wb") as handle:
            pickle.dump(self.tuned_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
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
