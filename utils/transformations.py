"""
Modulo para procesar y transformar datos.
Todas las clases tendrán al menos dos métodos:
   - fit() -> para ajustar los parámetros
   - transform() -> para transformar las features.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from skrub import GapEncoder
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ExtendedTransformation:

    def __init__(self, ge_components=50):
        self.imputer = SimpleImputer(strategy="median")
        self.ohEnconder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.gapEncoder = GapEncoder(n_components=ge_components)
        self.y_Transformer = QuantileTransformer()
        self.area_Transformer = QuantileTransformer()
        self.beds_Transformer = QuantileTransformer()
        self.polyfeatures = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        )
        self.scaler_y = StandardScaler()
        self.scaler_area = StandardScaler()
        self.scalar_beds = StandardScaler()

    def fit(self, X, y):
        X_data = X.copy()
        y_data = y.copy()
        print("X shape: ", X.shape)
        self.bin_vars_columns = X.columns[4:]
        print("bin_vars_columns shape: ", self.bin_vars_columns.shape)

        # fit impute n beds
        self.beds_feaures = "No. of Bedrooms"
        self.imputer.fit(X_data[[self.beds_feaures]])
        X_data = X_data.replace({9: np.nan})
        X_data[self.bin_vars_columns] = X_data[self.bin_vars_columns].replace(
            {0: "NO", 1: "SI", np.nan: "NO_DISPONIBLE"}
        )
        # fit low_cardinality features wiht ohot encoding
        self.low_card_columns = ["city"] + self.bin_vars_columns.to_list()
        print("low_card_columns shape: ", len(self.low_card_columns))  
        self.ohEnconder.fit(X_data[self.low_card_columns])
        self.loc_feature = "Location"

        # fit high_cardinality features.
        self.gapEncoder.fit(X_data[self.loc_feature])

        self.area_feature = "Area"

        # fit Quantile transformation of numerical vars.
        self.y_Transformer.fit(y_data)
        self.area_Transformer.fit(X_data[[self.area_feature]])
        self.beds_Transformer.fit(X_data[[self.beds_feaures]])

        self.scaler_y.fit(self.y_Transformer.transform(y_data))
        self.scaler_area.fit(
            self.area_Transformer.transform(X_data[[self.area_feature]])
        )
        self.scalar_beds.fit(
            self.beds_Transformer.transform(X_data[[self.beds_feaures]])
        )

        # scale to standard

    def transform(self, X_data, y_data):
        X = X_data.copy()
        y = y_data.copy()

        # impute missing data
        X = X.replace({9: np.nan})
        X[self.bin_vars_columns] = X[self.bin_vars_columns].replace(
            {0: "NO", 1: "SI", np.nan: "NO_DISPONIBLE"}
        )
        X[self.beds_feaures] = self.imputer.transform(X[[self.beds_feaures]])
        print("X shape: ", X.shape)
        # transform categorical features.
        cat_low_card_tfed = self.ohEnconder.transform(X[self.low_card_columns])
        X_low_card = pd.DataFrame(
            data=cat_low_card_tfed,
            columns=self.ohEnconder.get_feature_names_out(),
            index=X.index,
        )
        print("X_low_card   shape: ", X_low_card.shape)

        X_high_card = self.gapEncoder.transform(X[self.loc_feature])
        print("X_high_card shape: ", X_high_card.shape)

        # transform numerical vars.
        y_transformed = self.y_Transformer.transform(y)
        area_normal = self.area_Transformer.transform(X[[self.area_feature]])
        beds_normal = self.beds_Transformer.transform(X[[self.beds_feaures]])

        y_scaled = self.scaler_y.transform(y_transformed)
        area_scaled = self.scaler_area.transform(area_normal)
        beds_scaled = self.scalar_beds.transform(beds_normal)

        X_num = pd.DataFrame(
            data={
                self.area_feature: area_scaled.flatten(),
                self.beds_feaures: beds_scaled.flatten(),
            },
            index=X.index,
        )
        features_to_cross = pd.concat([X_low_card,X_num], axis=1)
        self.polyfeatures.fit(features_to_cross)
        crossed_features = self.polyfeatures.transform(features_to_cross)

        X_crossed_features = pd.DataFrame(
            data=crossed_features,
            columns=self.polyfeatures.get_feature_names_out(),
            index=X.index,
        )
        print("X_crossed_features shape: ", X_crossed_features.shape)
        X_EXPANDED = pd.concat([X_num, X_low_card, X_high_card, X_crossed_features], axis=1)
        print("X_EXPANDED shape: ", X_EXPANDED.shape)
        return X_EXPANDED, y_scaled

    def inverse_transform(self, y_data):
        return self.y_Transformer.inverse_transform(
            self.scaler_y.inverse_transform(y_data)
        )


class SimpleTransformation:

    def fit(self, X_data, y_data):
        self.remove_column = "Location"
        self.impute_columns = list(
            set(X_data.columns.to_list()) - set([self.remove_column, "city"])
        )
        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(X_data[self.impute_columns])

        self.ohEnconder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohEnconder.fit(X_data[["city"]])

    def transform(self, X_data, y_data):
        X = X_data.copy()
        y = y_data.copy()
        X = X.drop(columns=[self.remove_column])
        X[self.impute_columns] = self.imputer.transform(X[self.impute_columns])
        X_cat = pd.DataFrame(
            data=self.ohEnconder.transform(X_data[["city"]]),
            columns=self.ohEnconder.get_feature_names_out(),
            index=X.index,
        )
        X_final = pd.concat([X.drop(columns=["city"]), X_cat], axis=1)
        return X_final, y

class EncodedKNNTransformer:
    """
    Transformación personalizada:
    - Label Encoding para variables categóricas
    - Imputación con KNN
    - Clustering con KMeans (añade feature "Cluster")
    - Escalado de la variable objetivo
    """

    def __init__(self, n_neighbors=5, n_clusters=5):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_data = X.copy()
        y_data = y.copy()

        # Label Encoding
        self.categorical_columns = X_data.select_dtypes(include=['object']).columns
        for col in self.categorical_columns:
            le = LabelEncoder()
            X_data[col] = le.fit_transform(X_data[col].astype(str))
            self.label_encoders[col] = le

        # KMeans clustering
        self.kmeans.fit(X_data.select_dtypes(include=[np.number]))

        # KNN Imputer
        self.knn_imputer.fit(X_data)

        # Escalar la variable objetivo
        self.scaler.fit(y_data.values.reshape(-1, 1))

    def transform(self, X, y):
        X_data = X.copy()
        y_data = y.copy()

        # Label Encoding
        for col in self.categorical_columns:
            if col in X_data.columns:
                X_data[col] = self.label_encoders[col].transform(X_data[col].astype(str))

        # KNN imputación
        X_data = pd.DataFrame(self.knn_imputer.transform(X_data), columns=X.columns, index=X.index)

        # Clustering como nueva feature
        X_data['Cluster'] = self.kmeans.predict(X_data.select_dtypes(include=[np.number]))

        # Escalar y
        y_scaled = self.scaler.transform(y_data.values.reshape(-1, 1))

        return X_data, y_scaled

    def inverse_transform(self, y_scaled):
        return self.scaler.inverse_transform(y_scaled)
    
class FlaggedImputerTransformer:
    """
    Transformación personalizada:
    - Añade una nueva columna por cada feature con nulos, que indica si faltaba (1) o 
        no (0) → se llama por ejemplo "Area_missing_flag".
    - Rellena los nulos con la mediana guardada.
    - Si existe la columna "Area": crea una nueva columna "sqrt_area" con su raíz cuadrada.
    """

    def __init__(self):
        self.imputer_dict = {}  # Diccionario para guardar la mediana de cada feature
        self.features_to_transform = []  # Lista de columnas numéricas con missing

    def fit(self, X, y=None):
        # Identificar columnas numéricas con missing
        self.features_to_transform = [
            col for col in X.select_dtypes(include=[np.number]).columns
            if X[col].isnull().sum() > 0
        ]

        # Guardar la mediana para cada feature con missing
        for col in self.features_to_transform:
            self.imputer_dict[col] = X[col].median()

        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Para cada columna con missing:
        for col in self.features_to_transform:
            # Crear flag binario indicando si estaba missing
            X[col + "_missing_flag"] = X[col].isnull().astype(int)
            # Imputar con la mediana
            X[col] = X[col].fillna(self.imputer_dict[col])

        # Crear una nueva variable transformada a partir de 'Area'
        if "Area" in X.columns:
            X["sqrt_area"] = np.sqrt(X["Area"])

        return X, y
    
class AdvancedCategoricalTransformer:
    """
    Este transformador:
    - Agrupa categorías raras en 'Location' como 'Other'
    - Crea una nueva feature 'Location_freq' (frecuencia de cada location)
    - Hace target encoding sobre 'City' basado en Price
    """

    def __init__(self, rare_thresh=0.01):
        self.rare_thresh = rare_thresh
        self.city_price_map = {}
        self.location_freq = {}

    def fit(self, X, y):
        # Agrupar categorías raras en Location
        freq = X['Location'].value_counts(normalize=True)
        self.rare_locations = freq[freq < self.rare_thresh].index
        self.location_freq = X['Location'].value_counts()

        # Calcular target encoding para City
        self.city_price_map = y.groupby(X['city']).mean().to_dict()
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Agrupar locations raras
        X['Location_grouped'] = X['Location'].apply(lambda loc: 'Other' if loc in self.rare_locations else loc)

        # Codificar frecuencia de Location
        X['Location_freq'] = X['Location'].map(self.location_freq)
        X['Location_freq'] = X['Location_freq'].fillna(0)

        # Codificar City con target encoding
        X['City_encoded'] = X['city'].map(self.city_price_map)
        X['City_encoded'] = X['City_encoded'].fillna(np.median(list(self.city_price_map.values())))

        return X, y
