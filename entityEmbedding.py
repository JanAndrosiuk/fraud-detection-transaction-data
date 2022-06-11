import os
import pandas as pd
from sklearn.model_selection import train_test_split
# import tensorflow.python.layers.core
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.layers import Reshape, Embedding, BatchNormalization
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import re
import pickle


class EntityEmbeddings:
    def __init__(self, imp_datasets=None, imp_datasets_path=r"export/", pattern=r"df_train_imputed", load_pickle=False):

        self.target = ["isFraud"]
        self.cat_cols = [
            "DeviceInfo", "id_33", "id_31", "id_30", "R_emaildomain", "P_emaildomain",
            "ProductCD", "id_34", "card4", "M4", "id_23", "card6", "id_15", "id_37",
            "id_36", "id_38", "id_35", "DeviceType", "V65", "id_29", "id_28", "id_27",
            "V41", "V94", "id_16", "V88", "id_12", "V14"
        ]
        self.dataset_list = []

        # Load variables data types
        with open(imp_datasets_path+r"train_dtypes.pkl", 'rb') as handle:
            dtypes_dict = pickle.load(handle)

        # if imputed datasets were passed as the argument
        if imp_datasets is not None:
            for i in range(imp_datasets.dataset_count()):
                self.dataset_list.append(imp_datasets.complete_data(dataset=i, inplace=False).astype(dtypes_dict))

        # if imputed datasets should be loaded from drive
        elif not load_pickle and imp_datasets is None:
            # Search for imputed datasets
            path_list = os.listdir(imp_datasets_path)
            path_list = list(filter(re.compile(pattern).match, path_list))

            # Append those datasets to list
            for p in path_list:
                self.dataset_list.append(pd.read_csv(imp_datasets_path+p).astype(dtypes_dict))

        elif load_pickle and imp_datasets is None:
            path_list = os.listdir(imp_datasets_path)
            path_list = list(filter(re.compile(pattern).match, path_list))

            for p in path_list:
                # print(imp_datasets_path+p)
                with open(imp_datasets_path+p, "rb") as h:
                    df = pickle.load(h)
                df[self.cat_cols] = df[self.cat_cols].apply(lambda x: x.cat.codes)
                self.dataset_list.append(df)

        self.num_cols = [x for x in self.dataset_list[0].columns if x not in self.cat_cols+self.target]
        self.X_train_list, self.X_val_list, self.y_train_list, self.y_val_list = [], [], [], []
        self.emb_models = []

    def split_data(self):
        for df in self.dataset_list:
            split = train_test_split(
                df.drop(self.target, axis=1), df[self.target], test_size=0.33, random_state=2022, shuffle=False
            )
            self.X_train_list.append(split[0])
            self.X_val_list.append(split[1])
            self.y_train_list.append(split[2])
            self.y_val_list.append(split[3])

        return 0

    @staticmethod
    def get_emb_model(df, categorical_features, printable=False):

        inputs = []
        outputs = []

        for col in categorical_features:

            # find the cardinality of each categorical column, and set appropriate embedding dimension
            if printable:
                print(df[col].nunique(), df[col].max())
            cardinality = int(df[col].nunique()) + 1
            # cardinality = int(df[col].max())+1

            embedding_dim = max(min(cardinality//2, 100), 2)
            if printable:
                print(f'{col}: cardinality : {cardinality} and embedding dim: {embedding_dim}')

            inp = Input(shape=(1,))

            # Specify the embedding
            out = Embedding(
                cardinality, embedding_dim, input_shape=(1,), name=col+"_embed"
            )(inp)

            # out = SpatialDropout1D(0.1)(out)
            # out = Dropout(0.1)(out)

            # Flatten out the embeddings:
            out = Reshape(target_shape=(embedding_dim,))(out)

            # Add the input shape and embeddings to respective lists
            inputs.append(inp)
            outputs.append(out)

        # paste all the embeddings together
        x = Concatenate()(outputs)
        x = BatchNormalization()(x)

        # Add some general NN layers with dropout.
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)

        x = Dense(512, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)

        x = Dense(256, activation="sigmoid")(x)
        y_out = Dense(1)(x)

        # Specify and compile the model:
        embed_model = Model(inputs=inputs, outputs=y_out)
        if printable:
            print(embed_model.summary())

        return embed_model

    def embedding_preproc(self, X_train, X_val):
        """
        return lists with data for train, val and test set.
        Only categorical data, no numeric. (as we are just building the
        categorical embeddings)
        """
        input_list_train = []
        input_list_val = []

        for c in self.cat_cols:
            input_list_train.append(X_train[c].values.reshape(-1, 1))
            input_list_val.append(X_val[c].values.reshape(-1, 1))

        return input_list_train, input_list_val

    @staticmethod
    def auc(y_true, y_pred):
        def fallback_auc(y_true, y_pred):
            try:
                return roc_auc_score(y_true, y_pred)
            except:
                return 0.5
        return tensorflow.py_function(fallback_auc, (y_true, y_pred), tensorflow.double)

    def fit_model(self, save_dicts=True, verbose=1, printable=False):
        """
        https://stackoverflow.com/questions/62188532/invalidargumenterror-indices24-0-335-is-not-in-0-304-node-user-embed
        """

        for i in range(len(self.dataset_list)):

            # model = self.get_emb_model(np.vstack((self.X_train_list[i], self.X_val_list[i])), self.cat_cols)
            model = self.get_emb_model(
                pd.concat([self.X_train_list[i], self.X_val_list[i]], axis=0),
                self.cat_cols,
                printable=printable
            )
            # if printable:
            #     print(model.summary())

            model.compile(
                loss="binary_crossentropy",
                optimizer=Adam(learning_rate=0.00005),
                metrics=[self.auc]
            )

            # get the lists of data to feed into the Keras model:
            if printable:
                print("X train and validation shapes:", self.X_train_list[i].shape, self.X_val_list[i].shape)

            # Appending to train and validation lists
            X_embed_train, X_embed_val = self.embedding_preproc(self.X_train_list[i], self.X_val_list[i])

            es = EarlyStopping(
                monitor='val_auc', min_delta=0.001, patience=5, verbose=verbose, mode='max',
                baseline=None, restore_best_weights=True
            )

            rlr = ReduceLROnPlateau(
                monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6, mode='max', verbose=verbose
            )

            model.fit(
                X_embed_train,
                self.y_train_list[i].values,
                # utils.to_categorical(self.y_train_list[i].values),
                validation_data=(X_embed_val, self.y_val_list[i].values),
                callbacks=[es, rlr],
                batch_size=128, epochs=100, verbose=verbose
            )

            self.emb_models.append(model)

            embedding_dict = {}
            for c in self.cat_cols:

                embedding_dict[c] = model.get_layer(c + "_embed").get_weights()[0]

                if printable:
                    print(f"{c} dim: {len(embedding_dict[c][0])}")

            if save_dicts:
                if not os.path.exists(r"export"):
                    os.mkdir(r"export")
                with open(f"export/embedding_dict_{i}.pkl", 'wb') as handle:
                    pickle.dump(embedding_dict, handle)

        return 0

    def transform_cat_features(self, save_dataframe=True):
        for i in range(len(self.dataset_list)):

            with open(f"export/embedding_dict_{i}.pkl", 'rb') as handle:
                embedding_dict = pickle.load(handle)

            cat_emb_list = []
            for c in self.cat_cols:
                X = pd.concat([self.X_train_list[i], self.X_val_list[i]])
                cat_emb_list.append(
                    X[[c]]
                        .merge(pd.DataFrame(embedding_dict[c]), left_on=c, right_index=True)
                        .drop(c, axis=1)
                        .reset_index(drop=True)
                        .add_prefix(f"{c}_")
                )
                X_final = pd.concat(
                    [X.drop(self.cat_cols, axis=1), pd.concat(cat_emb_list, axis=1)],
                    axis=1
                )

                if save_dataframe:
                    if not os.path.exists(r"export"):
                        os.mkdir(r"export")
                    with open(f"export/X_final_{i}.pkl", "wb") as h:
                        pickle.dump(X_final, h, protocol=pickle.HIGHEST_PROTOCOL)

        return 0
