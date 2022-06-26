from src.setup_logger import *
from graphdatascience import GraphDataScience
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA
logger = logging.getLogger("PreprocessingElliptic")


class PreprocessingElliptic:
    def __init__(self,
                 neo4j_root_dir="C:/Users/user/.Neo4jDesktop/relate-data/dbmss/" +
                                "dbms-c3534c51-313d-4ff0-9094-0475b8061d1b/",
                 classes_path=r"../data/raw/elliptic_txs_classes.csv",
                 edgelist_path=r"../data/raw/elliptic_txs_edgelist.csv",
                 features_path=r"../data/raw/elliptic_txs_features.csv",
                 db_uri="bolt://localhost:7687", user="neo4j", password="password", graph_name="vesta",
                 node_label="node", rel_type="transaction"):
        logger.info("Loading Elliptic Dataset")
        df_classes = pd.read_csv(classes_path)
        df_features = pd.read_csv(features_path, header=None)
        self.df_nodes = df_classes.merge(df_features, left_on="txId", right_on=0)
        self.df_nodes["class"] = self.df_nodes["class"].replace({"unknown": 3}).astype("int64")
        self.df_nodes["label"] = "node"
        self.df_edges = pd.read_csv(edgelist_path)
        self.df_edges["type"] = "transaction"
        self.gds, self.db_uri, self.user, self.password, self.graph_name = None, db_uri, user, password, graph_name
        self.neo4j_root_dir, self.node_label, self.rel_type = neo4j_root_dir, node_label, rel_type
        self.train = None

    def export_neo4j(self):

        # Save nodes and edges prepared for Neo4j import
        self.df_nodes.to_csv(f"{self.neo4j_root_dir}import/nodes_neo4j.csv", header=False, index=False)
        self.df_edges.to_csv(f"{self.neo4j_root_dir}import/edges_neo4j.csv", header=False, index=False)

        # Save headers for nodes and edges dataframes prepared for Neo4j import
        feature_types = self.df_nodes.dtypes.replace({"int64": "int", "float64": "float", "object": "string"}).tolist()
        indices = self.df_nodes.dtypes.index
        feature_header = [f"{x}: {y}" for x, y in zip(indices, feature_types)]
        feature_header[0] = "id:ID"
        feature_header[-1] = ":LABEL"
        # noinspection PyTypeChecker
        np.savetxt(f"{self.neo4j_root_dir}import/nodes_header.csv", (feature_header, ), delimiter=",", fmt="% s")

        edges_header = self.df_edges.columns.tolist()
        edges_header = [f"{x}{y}" for x, y in zip(edges_header, [":START_ID", ":END_ID", ":TYPE"])]
        # noinspection PyTypeChecker
        np.savetxt(f"{self.neo4j_root_dir}import/edges_header.csv", (edges_header, ), delimiter=",", fmt="% s")

        return 0

    def create_graph(self):

        self.gds = GraphDataScience(self.db_uri, auth=(self.user, self.password))
        try:
            if self.gds.run_cypher(f"call gds.graph.exists('{self.graph_name}') yield exists").values[0]:
                logger.info("Graph already exists")
                return 0
            self.gds.run_cypher(
                f"""
                CALL gds.graph.project(
                '{self.graph_name}',
                '{self.node_label}',""" +
                "\n\t{%s: {orientation: 'UNDIRECTED'}})" % self.rel_type
            )
            logger.info("Graph successfully created")
        except Exception:
            raise Exception("wrong graph parameters / db server not started")

        logger.debug(self.gds.run_cypher(f"call gds.graph.exists('{self.graph_name}') yield exists"))

        return 0

    def prepare_train_set(self, erase_graph_df=False, save_target=True, save_target_name="elliptic_train_target.pkl"):

        self.df_edges = self.df_edges.merge(
            self.df_nodes.drop("label", axis=1).add_suffix("_from"), left_on="txId1", right_on="txId_from"
        )
        self.df_edges = self.df_edges.merge(
            self.df_nodes.drop("label", axis=1).add_suffix("_to"), left_on="txId2", right_on="txId_to"
        )
        self.df_edges.drop(["type", "txId1", "txId2", "txId_from", "txId_to"], axis=1, inplace=True)
        self.df_edges["target"] = self.df_edges[["class_from", "class_to"]].apply(
            lambda x: 1 if x["class_from"] == 1 or x["class_to"] == 1 else 0, axis=1
        )
        self.df_edges.drop(["class_from", "class_to"], axis=1, inplace=True)
        self.train = self.df_edges
        self.train.drop(["0_from", "0_to"], axis=1, inplace=True)

        if save_target:
            with open(f"../data/processed/{save_target_name}", "wb") as handle:
                pickle.dump(self.train["target"], handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.train.drop(["target"], axis=1, inplace=True)

        if erase_graph_df:
            self.df_nodes, self.df_edges = None, None

        return 0

    def pca(self, save_scaler=True, save_scaler_name="elliptic_scaler.pkl",
            pca_res_name="elliptic_train_pca.pkl",
            pca_model_name="pca_elliptic.pkl",
            save_pca_cols=True, pca_cols_name="elliptic_train_pca_cols.pkl"):

        sc = StandardScaler()
        self.train[self.train.columns] = sc.fit_transform(self.train[self.train.columns])
        if save_scaler:
            with open(f"../models/{save_scaler_name}", "wb") as handle:
                pickle.dump(sc, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pca = PCA(n_components=0.95)
        logger.info("Performing PCA on numerical variables")
        pca_res = pca.fit_transform(self.train)
        with open(f"../models/{pca_model_name}", "wb") as handle:
            pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"../data/processed/{pca_res_name}", "wb") as handle:
            pickle.dump(pca_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if save_pca_cols:
            cols = [f"pca_{x}" for x in range(pca_res.shape[1])]
            with open(f"../data/processed/{pca_cols_name}", "wb") as handle:
                pickle.dump(cols, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0
