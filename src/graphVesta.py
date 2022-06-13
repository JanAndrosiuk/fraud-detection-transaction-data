import numpy as np
import pandas as pd
from graphdatascience import GraphDataScience
import pickle
import re


class GraphVesta:
    def __init__(self, x_train_path="../data/processed/ieee_train_final_0.pkl",
                 y_train_path="../data/processed/ieee_train_y.pkl",
                 neo4j_root_dir="C:/Users/user/.Neo4jDesktop/relate-data/dbmss/" +
                                "dbms-c3534c51-313d-4ff0-9094-0475b8061d1b/",
                 raw_train_transaction_path="../data/raw/train_identity.csv",
                 raw_train_identity_path="../data/raw/train_identity.csv",
                 db_uri="bolt://localhost:7687", user="neo4j", password="password",
                 graph_name="elliptic", nodes_labels="['account', 'device']", rel_label="transaction"):
        self.x_train_path, self.y_train_path = x_train_path, y_train_path
        self.raw_train_transaction_path = raw_train_transaction_path,
        self.raw_train_identity_path = raw_train_identity_path
        self.neo4j_root_dir = neo4j_root_dir
        self.x_train, self.y_train = None, None
        self.train_nodes, self.train_edges = None, None
        self.signature_cols = ["start_node_id", "end_node_id"]
        self.gds, self.db_uri, self.user, self.password = None, db_uri, user, password
        self.graph_name, self.nodes_labels, self.rel_label = graph_name, nodes_labels, rel_label

    def load_raw_data(self):
        with open(self.x_train_path, "rb") as handle:
            self.x_train = pickle.load(handle)
        with open(self.y_train_path, "rb") as handle:
            self.y_train = pickle.load(handle)

        return 0

    def generate_signatures(self):
        cols = list(self.x_train.columns)
        start_node_sign_cols = list(filter(re.compile("card").match, cols))
        end_node_sign_cols = list(filter(re.compile("DeviceInfo").match, cols))

        start_node_signature = self.x_train.groupby(start_node_sign_cols).ngroup().values
        end_node_signature = self.x_train.groupby(end_node_sign_cols).ngroup().values

        self.x_train[self.signature_cols[0]] = start_node_signature
        self.x_train[self.signature_cols[0]] = "a"+self.x_train[self.signature_cols[0]].astype(str)

        self.x_train[self.signature_cols[1]] = end_node_signature
        self.x_train[self.signature_cols[1]] = "v"+self.x_train[self.signature_cols[1]].astype(str)

        return 0

    def prepare_neo4j_import(self, printable=False):
        """
        Divide variables into node variables and relationship variables
        :return: 0 if the process was completed successfully
        """

        # 1. Account nodes
        if printable:
            print("Preparing Account nodes")
        df_to_save = pd.DataFrame(self.x_train["start_node_id"].unique())
        df_to_save.columns = ["start_node_id:ID"]
        df_to_save["label:LABEL"] = "account"
        feature_header = list(df_to_save.columns)
        np.savetxt(
            self.neo4j_root_dir+"import/"+"nodes_account_header.csv", (feature_header,), delimiter=",", fmt="% s"
        )
        df_to_save.to_csv(
            self.neo4j_root_dir+"import/"+"nodes_account.csv", header=False, index=False
        )

        # 2. Device nodes
        if printable:
            print("Preparing Device nodes")
        df_to_save = pd.DataFrame(self.x_train["end_node_id"].unique())
        df_to_save.columns = ["end_node_id:ID"]
        df_to_save["label:LABEL"] = "device"
        feature_header = list(df_to_save.columns)
        np.savetxt(
            self.neo4j_root_dir+"import/"+"nodes_device_header.csv", (feature_header,), delimiter=",", fmt="% s"
        )
        df_to_save.to_csv(
            self.neo4j_root_dir+"import/"+"nodes_device.csv", header=False, index=False
        )

        # 3. Relationship edges
        if printable:
            print("Preparing Relationship nodes")
        dtype_neo4j_dict = {"int64": "int", "float64": "float", "float32": "float", "object": "string"}
        feature_dtypes = self.x_train.dtypes.replace(dtype_neo4j_dict)
        features = feature_dtypes.index
        dtypes = feature_dtypes.values
        feature_header = [str(x)+":"+y for x, y in zip(features, dtypes)]
        feature_header[-2] = "start_node_id:START_ID"
        feature_header[-1] = "end_node_id:END_ID"
        feature_header.extend(["type:TYPE"])
        df_to_save = self.x_train.copy()
        df_to_save["type"] = "transaction"
        np.savetxt(
            self.neo4j_root_dir+"import/"+"rels_header.csv", (feature_header,), delimiter=",", fmt="% s"
        )
        df_to_save.to_csv(
            self.neo4j_root_dir+"import/"+"rels.csv", header=False, index=False
        )

        return 0

    def create_bat_file(self):
        with open("../src/neo4j_ieee_import.bat", "w+") as h:
            h.write(
                f"cd {self.neo4j_root_dir}\n" + "bin/neo4j-admin.bat import " +
                "--nodes import/nodes_account_header.csv,import/nodes_account.csv " +
                "--nodes import/nodes_device_header.csv,import/nodes_device.csv " +
                "--relationships import/rels_header.csv,import/rels.csv " +
                "--force"
            )
        return 0

    def create_graph(self):
        self.gds = GraphDataScience(self.db_uri, auth=(self.user, self.password))
        try:
            self.gds.run_cypher(
                f"""
                CALL gds.graph.project(
                   '{self.graph_name}',
                   {self.nodes_labels},
                   '{self.rel_label}'
                )
                """
            )
            print("Graph successfully created")
        except Exception:
            raise Exception("Graph already exists / wrong graph parameters / db server not started")

        return 0

    def run(self, query=""):
        return self.gds.run_cypher(query)

    def append_metrics(self):

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
            yield nodeId, score as pagerank
            return pagerank
            """
        )
        eigenvector = self.run(
            f"""
            CALL gds.eigenvector.stream("{self.graph_name}")
            yield nodeId, score as eigenvector
            return eigenvector
            """
        )
        metrics_df = pd.concat(
            [degree, pagerank, eigenvector], axis=1
        )

        self.x_train = pd.concat([self.x_train, metrics_df], axis=1)

        return 0
