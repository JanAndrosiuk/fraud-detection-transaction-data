import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import miceforest
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle


class Preprocessing:

    def __init__(self, transactions_path="../data/raw/train_transaction.csv",
                 identity_path="../data/raw/train_identity.csv",
                 n_imp_datasets=3, target="isFraud", seed=2022):

        self.df_train_transaction_path = transactions_path
        self.df_train_identity_path = identity_path
        self.df_train = None
        self.nuniques = None
        self.vars_cat, self.vars_bool, self.vars_num = None, None, None
        self.encoder_dict = {}
        self.csfont = {"fontname": "Times New Roman"}
        self.seed = seed
        self.n_imp_datasets = n_imp_datasets  # number of imputed datasets
        self.target = target

    def load_merge_data(self, merge_on="TransactionID", printable=False):

        if printable:
            print("Loading and merging Transaction and Identity datasets")

        df_train_transaction = pd.read_csv(self.df_train_transaction_path)
        df_train_identity = pd.read_csv(self.df_train_identity_path)
        self.df_train = df_train_transaction.merge(df_train_identity, on=merge_on)

        return 0

    def delete_useless_train(self, printable=False):

        self.nuniques = self.df_train.nunique().sort_values(ascending=False)
        useless = self.nuniques.where(self.nuniques <= 1).dropna().index.values
        if printable:
            print("Variables which have 0 (only NAs) or 1 values (NA and one more values):", useless.shape[0])
        self.df_train.drop(axis=1, columns=useless, inplace=True, errors="ignore")

        cols_signature = ["TransactionID", "id_02"]
        self.df_train.drop(cols_signature, axis=1, inplace=True, errors="ignore")

        self.nuniques = self.df_train.nunique().sort_values(ascending=False)

        if printable:
            print("Shape after drop:", self.df_train.shape)

        return 0

    def plot_nans_var(self, save_plot=False):

        nan_sum = self.df_train.isna().sum()/self.df_train.shape[0]
        nan_sum.sort_values(ascending=False, inplace=True)
        nan_cols = nan_sum.index.values

        plt.figure(figsize=(20, 8))
        plt.bar(nan_cols, nan_sum)
        plt.xticks(rotation=90, fontsize=6)
        plt.xlabel("Training Set Variable", size=12, **self.csfont)
        plt.ylabel("Percentage of missingness", size=12, **self.csfont)
        plt.tick_params(
            axis='x', which='both', bottom=False, top=False, labelbottom=False
        )
        ax = plt.gca()
        ax.xaxis.grid(False)
        ax.locator_params(nbins=20, axis='y')
        plt.axhline(y=np.mean(nan_sum), color='r', linestyle="--", lw=1)
        if save_plot:
            if not os.path.exists("../reports/figures/"):
                os.mkdir("../reports/figures/")
            plt.savefig("../reports/figures/nan_var_ieee.png")
        plt.show()

        return 0

    def optimize_dtypes(self, printable=False, save_target=False):
        # Boolean variables have only 2 values
        self.vars_bool = list(self.nuniques.where(self.nuniques == 2).dropna().index.values)
        if printable:
            print("Boolean variables:", len(self.vars_bool))

        # Categorical variables are those which have more than 2 levels and are not floats
        self.vars_cat = self.nuniques.where(self.nuniques > 2).dropna().index.values
        self.vars_cat = [x for x in self.vars_cat if self.df_train[x].dtype not in ["float64", "int64"]]
        if printable:
            print("Categorical variables:", len(self.vars_cat))

        self.vars_num = [x for x in self.df_train.columns if x not in self.vars_cat+self.vars_bool]
        if printable:
            print("Numerical variables:", len(self.vars_num))

        # Cast categorical and boolean variables to pandas categorical data type
        self.df_train[self.vars_cat] = self.df_train[self.vars_cat].astype("category")
        self.df_train[self.vars_bool] = self.df_train[self.vars_bool].astype("category")

        # Optimize dtypes
        for col in self.df_train.columns:
            if self.df_train[col].dtype == "float64":
                self.df_train[col] = self.df_train[col].astype('float32')
            if self.df_train[col].dtype == "int64":
                self.df_train[col] = self.df_train[col].astype('int32')

        # Lower Transaction Date values
        self.df_train["TransactionDT"] = self.df_train["TransactionDT"]/86400

        # Save target vector
        if save_target:
            if not os.path.exists("../data/processed/"):
                os.mkdir("../data/processed/")
            with open("../data/processed/ieee_train_y.pkl", 'wb') as handle:
                pickle.dump(self.df_train[self.target].values, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0

    def label_encoding(self, save_encodings=False, save_dtypes=False):

        # Fit the encoder of target labels (from string to numeric)
        for col in self.vars_cat:
            self.encoder_dict[col] = LabelEncoder().fit(self.df_train[col])
        for col in self.vars_bool:
            self.encoder_dict[col] = LabelEncoder().fit(self.df_train[col])

        # Transform dataset variables
        for col in self.vars_cat:
            self.df_train[col] = self.encoder_dict[col].transform(self.df_train[col])
        for col in self.vars_bool:
            self.df_train[col] = self.encoder_dict[col].transform(self.df_train[col])

        # To categorical
        self.df_train[self.vars_cat+self.vars_bool] = \
            self.df_train[self.vars_cat+self.vars_bool].astype("category")

        # Save dictionary with encodings
        if save_dtypes:
            if not os.path.exists("../data/interim/"):
                os.mkdir("../data/interim/")
            with open("../data/interim/ieee_train_dtypes.pkl", 'wb') as handle:
                pickle.dump(self.df_train.dtypes.to_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save label encodings
        if save_encodings:
            if not os.path.exists("../data/interim/"):
                os.mkdir("../data/interim/")
            with open("../data/interim/ieee_train_label_encodings.pkl", "wb") as handle:
                pickle.dump(self.encoder_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0

    def iteratively_impute(self, max_iter=10, verbose=0, save_dataset=True, printable=False):
        imp_num = IterativeImputer(
            # estimator=RandomForestRegressor(),
            initial_strategy='mean', max_iter=max_iter, random_state=self.seed,
            verbose=verbose
        )
        # imp_cat = IterativeImputer(
        #     estimator=RandomForestClassifier(),
        #     initial_strategy='most_frequent', max_iter=max_iter, random_state=self.seed,
        #     verbose=verbose
        # )
        if printable:
            print("Imputing numerical data")
        self.df_train[self.vars_num] = imp_num.fit_transform(self.df_train[self.vars_num])
        # if printable: print("Imputing categorical data")
        # self.df_train[self.vars_cat+self.vars_bool] = \
        #     imp_cat.fit_transform(self.df_train[self.vars_cat+self.vars_bool])

        # Save imputed dataset
        if save_dataset:
            if printable:
                print("Saving imputed data")
            if not os.path.exists("../data/interim/"):
                os.mkdir("../data/interim/")
            with open("../data/interim/ieee_train_imputed.pkl", 'wb') as handle:
                pickle.dump(self.df_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0

    def impute_miceforest(self, printable=False, save_datasets=True, return_datasets=False):
        # https://morioh.com/p/e19cd87c66e3

        amp = miceforest.ampute_data(self.df_train, perc=0.25, random_state=self.seed)

        kds = miceforest.ImputationKernel(
            amp,
            datasets=self.n_imp_datasets,
            save_all_iterations=True,
            # categorical_feature=self.vars_cat+self.vars_bool,
            categorical_feature="auto",
            random_state=self.seed
        )
        if printable:
            print(kds)

        kds.mice(2)

        if save_datasets:
            if not os.path.exists("../data/interim/"):
                os.mkdir("../data/interim/")
            for i in range(self.n_imp_datasets):
                completed_dataset = kds.complete_data(dataset=i, inplace=False)
                completed_dataset.to_csv(f"../data/interim/ieee_train_mf_imputed_{i}.csv", index=False)

        if return_datasets:
            return kds

        return 0
