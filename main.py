from src.preprocessing import *
from src.entityEmbedding import *
from src.randomForest import *


def main():
    preprocessing = Preprocessing()
    preprocessing.load_merge_data(printable=True)
    preprocessing.delete_useless_train(printable=True)
    preprocessing.plot_nans_var(save_plot=True)
    preprocessing.optimize_dtypes(printable=True)
    preprocessing.label_encoding(save_encodings=True, save_dtypes=True)
    preprocessing.iteratively_impute(max_iter=10, verbose=2, printable=True)

    em = EntityEmbeddings(load_pickle=True)
    em.split_data()
    em.fit_model(verbose=2, printable=True)
    em.transform_cat_features()

    rf = RF(n_jobs=10)
    rf.load_data()
    rf.cv_base_model(verbose=2)
    rf.tune_model(verbose=2, n_param_samples=25)
    rf.save_tuned_models()

    return 0


if __name__ == "__main__":
    main()
