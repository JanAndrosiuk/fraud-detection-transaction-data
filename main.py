from src.preprocessing_vesta import *
from src.entity_embedding_vesta import *
from src.random_forest_vesta import *
from src.graph_vesta import *
from src.preprocessing_elliptic import *


def main():
    preprocessing = Preprocessing()
    preprocessing.load_merge_data()
    preprocessing.delete_useless_train()
    preprocessing.plot_nans_var(save_plot=True)
    preprocessing.plot_missing_patterns(save_fig=True)
    preprocessing.optimize_dtypes(save_target=True, save_cat_vars=True)
    preprocessing.label_encoding(save_encodings=True, save_dtypes=True)
    preprocessing.iteratively_impute(max_iter=10, verbose=2)
    preprocessing.pca(save_scaler=True, merge_with_mca=True)

    em = EntityEmbeddings(load_pickle=True)
    em.split_data()
    em.fit_model(verbose=2)
    em.transform_cat_features()

    rf = RF(n_jobs=10)
    rf.load_data()
    rf.cv_base_model(verbose=2)
    rf.tune_model(verbose=2, n_param_samples=25)

    gv = GraphVesta()
    gv.load_raw_data()
    gv.generate_signatures()
    gv.prepare_neo4j_import()
    gv.create_bat_file()
    gv.append_metrics(mono=False, save_dataset=True, delete_sign=False)
    gv.one_mode_projection()
    gv.prepare_mono_neo4j_import(create_bat=True)
    gv.create_mono_graph()
    gv.append_metrics(mono=True, save_dataset=True, delete_sign=False)

    prel = PreprocessingElliptic()
    prel.export_neo4j()
    prel.prepare_train_set(erase_graph_df=True)
    prel.create_graph()

    return 0


if __name__ == "__main__":
    main()
