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
                 mono_label="account", save_prefix="vesta_"):
        self.x_train_path, self.y_train_path = x_train_path, y_train_path
        self.raw_imputed_path, self.neo4j_root_dir = raw_imputed_path, neo4j_root_dir
        self.x_train, self.y_train, self.train_nodes, self.train_edges = None, None, None, None
        self.signature_cols, self.metrics_cols = ["start_node_id", "end_node_id"], None
        self.gds, self.db_uri, self.user, self.password = None, db_uri, user, password
        self.graph_name, self.nodes_labels, self.rel_label = graph_name, nodes_labels, rel_label
        self.mono_graph_name, self.mono_label, self.monopartite = f"mono_{self.graph_name}", mono_label, None
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
        # logger.info(f"Unique account ids: {start_node_signature.shape},
        # unique device ids: {end_node_signature.shape}")

        df_res = pd.DataFrame()
        df_res[self.signature_cols[0]] = start_node_signature
        df_res[self.signature_cols[0]] = "a"+df_res[self.signature_cols[0]].astype(str)
        df_res[self.signature_cols[1]] = end_node_signature
        df_res[self.signature_cols[1]] = "v"+df_res[self.signature_cols[1]].astype(str)

        self.x_train = pd.concat([self.x_train, df_res], axis=1)

        return 0

    def prepare_neo4j_import(self, create_bat=True):
        """
        Divide variables into node variables and relationship variables
        :return: 0 if the process was completed successfully
        """
        # Transform variables with category dtype back to int32:
        cat_dtype_cols = [x for x in self.x_train.columns if self.x_train[x].dtype == "category"]
        self.x_train[cat_dtype_cols] = self.x_train[cat_dtype_cols].astype(int)

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
        dtype_neo4j_dict = {"int64": "int", "int32": "int", "float64": "float", "float32": "float", "object": "string"}
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

        if create_bat:
            self.create_bat_file()

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

    def append_metrics(self, save_dataset=True, delete_sign=False, mono=False):

        init_cols = self.x_train.columns
        if mono:
            graph_name = self.mono_graph_name
            self.create_mono_graph()
        else:
            graph_name = self.graph_name
            self.create_graph()

        # Calculate centrality metrics
        if mono:
            logger.info("Calculating degree")
            degree = self.run(
                f"""
                call gds.degree.stream("{graph_name}")
                yield nodeId, score as degree
                return coalesce(gds.util.asNode(nodeId).Id, "") as id, degree
                """
            )
        else:
            logger.info("Calculating degree")
            degree = self.run(
                f"""
            call gds.degree.stream("{self.graph_name}")
            yield nodeId, score as degree
            return coalesce(gds.util.asNode(nodeId).{self.signature_cols[0]}, "") + 
                coalesce(gds.util.asNode(nodeId).{self.signature_cols[1]}, "") as id, degree
            """
            )
        logger.info("Calculating pagerank")
        pagerank = self.run(
            f"""
            CALL gds.pageRank.stream("{graph_name}")
            yield nodeId, score as pagerank return pagerank
            """
        )
        logger.info("Calculating hits scores")
        hits = self.run(
            f"""
            CALL gds.alpha.hits.stream("{graph_name}", {{hitsIterations: 20}})
            yield nodeId, values
            return values.auth as hits_auth, values.hub as hits_hub
            """
        )
        logger.info("Calculating betweenness")
        betweenness = self.run(
            f"""
            CALL gds.betweenness.stream("{graph_name}")
            yield nodeId, score as betweenness return betweenness
            """
        )
        logger.info("Calculating wcc size")
        wcc_size = self.run(
            f"""
            CALL gds.wcc.stream("{graph_name}")
            yield nodeId, componentId
            """
        )
        logger.info("Calculating louvain community sizes")
        louvain_size = self.run(
            f"call gds.louvain.stream('{graph_name}') yield nodeId, communityId"
        )

        # Calculate size of the WCC
        logger.debug("Extracting wcc info")
        map_dict = wcc_size.groupby("componentId").size().to_frame()
        map_dict.columns = ["wcc_size"]
        wcc_size = wcc_size.merge(map_dict, left_on="componentId", right_index=True) \
            .drop(["nodeId", "componentId"], axis=1)

        # Calculate size of the community each node belongs to
        logger.debug("Extracting louvain communities' info")
        map_dict = louvain_size.groupby("communityId").size().to_frame()
        map_dict.columns = ["community_size"]
        louvain_size = louvain_size.merge(map_dict, left_on="communityId", right_index=True) \
            .drop(["nodeId", "communityId"], axis=1)

        # Collect all metrics in one dataframe describing each node
        logger.debug("Concatenating metrics")
        metrics_df = pd.concat(
            [degree, pagerank, hits, betweenness, wcc_size, louvain_size], axis=1
        )
        if mono:
            logger.info("Merging and saving training data set with monopartite graph features")
            self.x_train = self.x_train.merge(
                metrics_df, how="left", left_on="start_node_id", right_on="id"
            )
            if delete_sign:
                self.x_train.drop(["start_node_id", "end_node_id", "id"], axis=1, inplace=True)
            self.metrics_cols = [x for x in list(self.x_train.columns) if x not in init_cols]
            logger.debug(f"Metric columns: {self.metrics_cols}")

        else:
            logger.info("Merging and saving training data set with bipartite graph features")
            metrics_acc = metrics_df.loc[metrics_df["id"].str[0] == "a"].add_prefix("acc_")
            metrics_dev = metrics_df.loc[metrics_df["id"].str[0] == "v"].add_prefix("dev_")

            # Merge calculated metrics with train dataset
            self.x_train = self.x_train.merge(
                metrics_acc, how="left", left_on="start_node_id", right_on="acc_id"
            )
            self.x_train = self.x_train.merge(
                metrics_dev, how="left", left_on="end_node_id", right_on="dev_id"
            )
            if delete_sign:
                self.x_train.drop(["start_node_id", "end_node_id", "acc_id", "dev_id"], axis=1, inplace=True)
            else:
                self.x_train.drop(["acc_id", "dev_id"], axis=1, inplace=True)

            self.metrics_cols = [x for x in list(self.x_train.columns) if x not in init_cols]
            logger.debug(f"Metric columns: {self.metrics_cols}")

        if save_dataset:
            logger.info("Saving data")
            if not os.path.exists("../data/processed"):
                os.mkdir("../data/processed")
            if mono:
                with open(f"../data/processed/mono_{self.save_prefix}graph_train_0.pkl", "wb") as handle:
                    pickle.dump(self.x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(f"../data/processed/{self.save_prefix}graph_train_0.pkl", "wb") as handle:
                    pickle.dump(self.x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0

    def one_mode_projection(self):
        """
        Equivalent in Neo4j would be something like (but it takes too long):
            MATCH (a1:account)-[*2]-(a2:account)
            WHERE id(a1) < id(a2)
            WITH a1, a2, count(*) as strength
            CREATE (a1)-[r:RELATED_TO]->(a2)
            SET r.strength = strength
        :return:
        """

        # Take only start node id (account id) and end node id (device id)
        adj_df = self.x_train[["start_node_id", "end_node_id"]].copy()

        # buffer column which will be used for aggregation purposes to indicate presence of connection
        adj_df["exist"] = 1

        # Turn the dataframe to pivot form which indicates adjacency between account and device nodes
        adj_df = pd.pivot_table(
            adj_df, values="exist", index=["start_node_id"], columns=["end_node_id"], aggfunc=np.sum
        )

        # NAs have to be temporarily converted to 0 in order to allow multiplication
        adj_df.fillna(0, inplace=True)

        # Convert to numpy matrix form for the ease of calculations
        adj_matrix = adj_df.values

        # Compute the dot product of adjacency matrix and its transposition which is the core calculation of
        # the projection from bipartite to monopartite graph
        adj_matrix = adj_matrix.dot(adj_matrix.T)

        # recurrent links are not allowed, therefore diagonal should be filled with zeros
        np.fill_diagonal(adj_matrix, 0)

        # only one way connections relevant, therefore
        # only the upper (or lower, doesn't matter) diagonal is analyzed
        adj_matrix *= 1 - np.tri(*adj_matrix.shape, k=-1)

        # converting NAs back to zero to allow the later conversion method (to pandas DataFrame) to skip them
        adj_matrix[adj_matrix == 0] = np.nan

        # stack method transforms the DataFrame to long form with multi-index
        monopartite = pd.DataFrame(
            adj_matrix, index=adj_df.index, columns=adj_df.index
        ).stack()
        monopartite.index.names = ["start", "end"]

        # resetting the index yields the clean form of graph representation which is ready to import to Neo4j
        monopartite = monopartite.reset_index()
        monopartite.rename(columns={0: "strength"}, inplace=True)
        monopartite["strength"] = monopartite["strength"].astype(int)

        self.monopartite = monopartite

        return 0

    def prepare_mono_neo4j_import(self, create_bat=True):
        """
        Divide variables into node variables and relationship variables
        :return: 0 if the process was completed successfully
        """

        # 1. Account nodes
        logger.info("Preparing and saving Account nodes")
        df_to_save = pd.DataFrame(
            {"Id:ID": self.x_train["start_node_id"].unique(), "label:LABEL": "account"}
        )
        feature_header = list(df_to_save.columns)
        # noinspection PyTypeChecker
        np.savetxt(
            f"{self.neo4j_root_dir}import/mono_{self.save_prefix}nodes_header.csv",
            (feature_header,), delimiter=",", fmt="% s"
        )
        df_to_save.to_csv(
            f"{self.neo4j_root_dir}import/mono_{self.save_prefix}nodes.csv", header=False, index=False
        )

        # 2. Relationship edges
        logger.info("Preparing and saving Relationship nodes")
        df_to_save = self.monopartite.copy()
        df_to_save.columns = ["start:START_ID", "end:END_ID", "strength:int"]
        df_to_save["type:TYPE"] = self.rel_label
        feature_header = list(df_to_save.columns)
        # noinspection PyTypeChecker
        np.savetxt(
            f"{self.neo4j_root_dir}import/mono_{self.save_prefix}rels_header.csv",
            (feature_header,), delimiter=",", fmt="% s"
        )
        df_to_save.to_csv(
            f"{self.neo4j_root_dir}import/mono_{self.save_prefix}rels.csv", header=False, index=False
        )

        if create_bat:
            with open(f"../src/neo4j_mono_{self.save_prefix}import.bat", "w+") as h:
                h.write(
                    f"cd {self.neo4j_root_dir}\n" + "bin/neo4j-admin.bat import " +
                    f"--nodes import/mono_{self.save_prefix}nodes_header.csv," +
                    f"import/mono_{self.save_prefix}nodes.csv " +
                    f"--relationships import/mono_{self.save_prefix}rels_header.csv," +
                    f"import/mono_{self.save_prefix}rels.csv " +
                    "--force"
                )

        return 0

    def create_mono_graph(self):

        self.gds = GraphDataScience(self.db_uri, auth=(self.user, self.password))
        try:
            if self.run(f"call gds.graph.exists('{self.mono_graph_name}') yield exists").values[0]:
                logger.info("Graph already exists")
                return 0
            self.gds.run_cypher(
                f"""
                CALL gds.graph.project(
                '{self.mono_graph_name}',
                '{self.mono_label}',""" +
                "\n\t{%s: {orientation: 'UNDIRECTED'}})" % self.rel_label
            )
            logger.info("Graph successfully created")
        except Exception:
            raise Exception("wrong graph parameters / db server not started")

        logger.debug(self.run(f"call gds.graph.exists('{self.mono_graph_name}') yield exists"))

        return 0
