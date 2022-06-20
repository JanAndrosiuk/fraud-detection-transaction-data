from src.setup_logger import *
import numpy as np
import pandas as pd
from graphdatascience import GraphDataScience
import pickle
import re
import os
logger = logging.getLogger("VestaGraph")


class GraphVesta:
    def __init__(self, x_train_path="../data/processed/ieee_train_final_0.pkl",
                 y_train_path="../data/processed/ieee_train_y.pkl",
                 neo4j_root_dir="C:/Users/user/.Neo4jDesktop/relate-data/dbmss/" +
                                "dbms-c3534c51-313d-4ff0-9094-0475b8061d1b/",
                 raw_imputed_path="../data/interim/ieee_train_imputed_old.pkl",
                 db_uri="bolt://localhost:7687", user="neo4j", password="password",
                 graph_name="vesta", nodes_labels="['account', 'device']", rel_label="transaction",
                 save_prefix="vesta_"):
        self.x_train_path, self.y_train_path = x_train_path, y_train_path
        self.raw_imputed_path = raw_imputed_path
        self.neo4j_root_dir = neo4j_root_dir
        self.x_train, self.y_train = None, None
        self.train_nodes, self.train_edges = None, None
        self.signature_cols, self.metrics_cols = ["start_node_id", "end_node_id"], None
        self.gds, self.db_uri, self.user, self.password = None, db_uri, user, password
        self.graph_name, self.nodes_labels, self.rel_label = graph_name, nodes_labels, rel_label
        self.save_prefix = save_prefix

    def load_raw_data(self):
        with open(self.x_train_path, "rb") as handle:
            self.x_train = pickle.load(handle)
        with open(self.y_train_path, "rb") as handle:
            self.y_train = pickle.load(handle)
        logger.debug(f"Data set shapes -> x_train: {self.x_train.shape}, y_train: {self.y_train.shape}")

        return 0

    def generate_signatures(self):

        with open(self.raw_imputed_path, "rb") as h:
            train_imp = pickle.load(h)

        cols = list(train_imp.columns)
        start_node_sign_cols = list(filter(re.compile("card").match, cols))
        logger.debug(f"Columns used to create signature: {start_node_sign_cols}")
        end_node_sign_cols = list(filter(re.compile("DeviceInfo").match, cols))

        start_node_signature = train_imp.groupby(start_node_sign_cols).ngroup().values
        end_node_signature = train_imp.groupby(end_node_sign_cols).ngroup().values

        df_res = pd.DataFrame()
        df_res[self.signature_cols[0]] = start_node_signature
        df_res[self.signature_cols[0]] = "a"+df_res[self.signature_cols[0]].astype(str)
        df_res[self.signature_cols[1]] = end_node_signature
        df_res[self.signature_cols[1]] = "v"+df_res[self.signature_cols[1]].astype(str)

        self.x_train = pd.concat([self.x_train, df_res], axis=1)

        return 0

    def prepare_neo4j_import(self):
        """
        Divide variables into node variables and relationship variables
        :return: 0 if the process was completed successfully
        """

        # 1. Account nodes
        logger.info("Preparing and saving Account nodes")
        df_to_save = pd.DataFrame(self.x_train["start_node_id"].unique())
        df_to_save.columns = ["start_node_id:ID"]
        df_to_save["label:LABEL"] = "account"
        feature_header = list(df_to_save.columns)
        # noinspection PyTypeChecker
        np.savetxt(
            f"{self.neo4j_root_dir}import/{self.save_prefix}nodes_account_header.csv",
            (feature_header,), delimiter=",", fmt="% s"
        )
        df_to_save.to_csv(
            f"{self.neo4j_root_dir}import/{self.save_prefix}nodes_account.csv", header=False, index=False
        )

        # 2. Device nodes
        logger.info("Preparing and saving Device nodes")
        df_to_save = pd.DataFrame(self.x_train["end_node_id"].unique())
        df_to_save.columns = ["end_node_id:ID"]
        df_to_save["label:LABEL"] = "device"
        feature_header = list(df_to_save.columns)
        # noinspection PyTypeChecker
        np.savetxt(
            f"{self.neo4j_root_dir}import/{self.save_prefix}nodes_device_header.csv",
            (feature_header,), delimiter=",", fmt="% s"
        )
        df_to_save.to_csv(
            f"{self.neo4j_root_dir}import/{self.save_prefix}nodes_device.csv", header=False, index=False
        )

        # 3. Relationship edges
        logger.info("Preparing and saving Relationship nodes")
        dtype_neo4j_dict = {"int64": "int", "float64": "float", "float32": "float", "object": "string"}
        feature_dtypes = self.x_train.dtypes.replace(dtype_neo4j_dict)
        features = feature_dtypes.index
        dtypes = feature_dtypes.values
        feature_header = [f"{x}:{y}" for x, y in zip(features, dtypes)]
        feature_header[-2] = "start_node_id:START_ID"
        feature_header[-1] = "end_node_id:END_ID"
        feature_header.extend(["type:TYPE"])
        df_to_save = self.x_train.copy()
        df_to_save["type"] = "transaction"
        # noinspection PyTypeChecker
        np.savetxt(
            f"{self.neo4j_root_dir}import/{self.save_prefix}rels_header.csv",
            (feature_header,), delimiter=",", fmt="% s"
        )
        df_to_save.to_csv(
            f"{self.neo4j_root_dir}import/{self.save_prefix}rels.csv", header=False, index=False
        )

        return 0

    def create_bat_file(self):
        with open(f"../src/neo4j_{self.save_prefix}import.bat", "w+") as h:
            h.write(
                f"cd {self.neo4j_root_dir}\n" + "bin/neo4j-admin.bat import " +
                f"--nodes import/{self.save_prefix}nodes_account_header.csv," +
                f"import/{self.save_prefix}nodes_account.csv " +
                f"--nodes import/{self.save_prefix}nodes_device_header.csv," +
                f"import/{self.save_prefix}nodes_device.csv " +
                f"--relationships import/{self.save_prefix}rels_header.csv,import/{self.save_prefix}rels.csv " +
                "--force"
            )
        return 0

    def create_graph(self):
        self.gds = GraphDataScience(self.db_uri, auth=(self.user, self.password))
        try:
            if self.run(f"call gds.graph.exists('{self.graph_name}') yield exists").values[0]:
                print("Graph already exists")
                return 0
            self.gds.run_cypher(
                f"""
                CALL gds.graph.project(
                   '{self.graph_name}',
                   {self.nodes_labels},
                   '{self.rel_label}'
                )
                """
            )
            logger.info("Graph successfully created")
        except Exception:
            raise Exception("wrong graph parameters / db server not started")

        return 0

    def run(self, query=""):
        return self.gds.run_cypher(query)

    def append_metrics(self, save_dataset=True):

        init_cols = self.x_train.columns

        # Calculate centrality metrics
        degree = self.run(
            f"""
            call gds.degree.stream("{self.graph_name}")
            yield nodeId, score as degree
            return coalesce(gds.util.asNode(nodeId).{self.signature_cols[0]}, "") + 
                coalesce(gds.util.asNode(nodeId).{self.signature_cols[1]}, "") as id, degree
            """
        )
        pagerank = self.run(
            f"""
            CALL gds.pageRank.stream("{self.graph_name}")
            yield nodeId, score as pagerank return pagerank
            """
        )
        hits = self.run(
            f"""
            CALL gds.alpha.hits.stream("{self.graph_name}", {{hitsIterations: 20}})
            yield nodeId, values
            return values.auth as hits_auth, values.hub as hits_hub
            """
        )
        betweenness = self.run(
            f"""
            CALL gds.betweenness.stream("{self.graph_name}")
            yield nodeId, score as betweenness return betweenness
            """
        )
        wcc_size = self.run(
            f"""
            CALL gds.wcc.stream("{self.graph_name}")
            yield nodeId, componentId
            """
        )
        louvain_size = self.run(
            f"call gds.louvain.stream('{self.graph_name}') yield nodeId, communityId"
        )

        # Calculate size of the WCC
        map_dict = wcc_size.groupby("componentId").size().to_frame()
        map_dict.columns = ["wcc_size"]
        wcc_size = wcc_size.merge(map_dict, left_on="componentId", right_index=True) \
            .drop(["nodeId", "componentId"], axis=1)

        # Calculate size of the community each node belongs to
        map_dict = louvain_size.groupby("communityId").size().to_frame()
        map_dict.columns = ["community_size"]
        louvain_size = louvain_size.merge(map_dict, left_on="communityId", right_index=True) \
            .drop(["nodeId", "communityId"], axis=1)

        # Collect all metrics in one dataframe describing each node
        metrics_df = pd.concat(
            [degree, pagerank, hits, betweenness, wcc_size, louvain_size], axis=1
        )
        metrics_acc = metrics_df.loc[metrics_df["id"].str[0] == "a"].add_prefix("acc_")
        metrics_dev = metrics_df.loc[metrics_df["id"].str[0] == "v"].add_prefix("dev_")

        # Merge calculated metrics with train dataset
        self.x_train = self.x_train.merge(
            metrics_acc, how="left", left_on="start_node_id", right_on="acc_id"
        )
        self.x_train = self.x_train.merge(
            metrics_dev, how="left", left_on="end_node_id", right_on="dev_id"
        )
        self.x_train.drop(["start_node_id", "end_node_id", "acc_id", "dev_id"], axis=1, inplace=True)

        self.metrics_cols = [x for x in list(self.x_train.columns) if x not in init_cols]
        logger.debug(f"Metric columns: {self.metrics_cols}")

        if save_dataset:
            if not os.path.exists("../data/processed"):
                os.mkdir("../data/processed")
            with open(f"../data/processed/{self.save_prefix}graph_train_0.pkl", "wb") as handle:
                pickle.dump(self.x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0
