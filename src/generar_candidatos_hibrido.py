import os, random, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from lightgbm import LGBMClassifier
from implicit.als import AlternatingLeastSquares
import scipy.sparse as sparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

SEED = 42
def set_all_seeds(seed=42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed); np.random.seed(seed)
set_all_seeds(SEED)

# ---------- 1. Datos (identico origen que usaron ambas companeras) ----------
interacciones_df = pd.read_csv("data/processed/interacciones_clean.csv", parse_dates=["order_date"])
product_df = pd.read_csv("data/processed/product_clean.csv")
print(f"Productos: {product_df.shape[0]} | Interacciones: {interacciones_df.shape[0]}")

# ---------- 2. Componente de CONTENIDO (tal cual el notebook de la companera) ----------
product_model = product_df.copy()
cat = product_model["product_category"].fillna("").astype(str).str.lower().str.strip()
subcat = product_model["product_subcategory"].fillna("").astype(str).str.lower().str.strip()
brand = product_model["brand"].fillna("").astype(str).str.lower().str.strip()
product_model["metadata_text"] = cat + " " + subcat + " " + brand

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(product_model["metadata_text"])

scaler = MinMaxScaler()
numeric_features = scaler.fit_transform(product_model[["unit_price", "product_rating"]].fillna(0))
numeric_features_weighted = numeric_features * 0.2

item_features_matrix = sp.hstack([tfidf_matrix, numeric_features_weighted], format="csr")
cosine_sim_matrix = cosine_similarity(item_features_matrix, item_features_matrix)
print(f"Matriz de atributos: {item_features_matrix.shape} | Similitud coseno: {cosine_sim_matrix.shape}")

index_product = pd.Series(product_model.index, index=product_model["product_id"]).drop_duplicates()

# ---------- 3. Split temporal GLOBAL (identico al de la companera) ----------
interacciones_model = interacciones_df.sort_values(by=["order_date", "customer_id", "product_id"]).reset_index(drop=True)
split_idx = int(len(interacciones_model) * 0.8)
train_df = interacciones_model.iloc[:split_idx]
test_df = interacciones_model.iloc[split_idx:]
print(f"Train: {len(train_df)} | Test: {len(test_df)}")


def evaluation_contend(customer_id, df_train, k_top=10, alpha_recompra=1.5):
    """Funcion de generacion de candidatos - identica a la del notebook de la companera."""
    customer_history = (
        df_train[df_train["customer_id"] == customer_id]
        .groupby("product_id")["quantity"].sum().reset_index()
    )
    if customer_history.empty:
        return []
    valid_history = customer_history[customer_history["product_id"].isin(index_product.index)].copy()
    if valid_history.empty:
        return []
    valid_history["idx"] = valid_history["product_id"].map(index_product)
    indices = valid_history["idx"].values
    quantities = valid_history["quantity"].values

    user_vectors = item_features_matrix[indices].toarray()
    weighted_vectors = user_vectors * quantities[:, np.newaxis]
    user_profile_vector = weighted_vectors.sum(axis=0) / quantities.sum()
    content_scores = cosine_similarity(user_profile_vector.reshape(1, -1), item_features_matrix)[0]

    repurchase_scores = np.zeros(len(product_model))
    for idx_item, qty in zip(indices, quantities):
        repurchase_scores[idx_item] = np.log1p(qty)

    final_scores = content_scores + (alpha_recompra * repurchase_scores)
    top_indices = np.argsort(final_scores)[::-1][:k_top]
    return [product_model.iloc[i]["product_id"] for i in top_indices]


# ---------- 4. Componente COLABORATIVO (ALS, mismos hiperparametros que el Modelo 1) ----------
# Se re-entrena sobre el MISMO split temporal que usa el modelo de contenido, para que
# el score de ALS sea una feature valida y alineada con el resto del dataset del hibrido.
df_als = train_df.groupby(["customer_id", "product_id"])["quantity"].sum().reset_index()
df_als["customer_id"] = df_als["customer_id"].astype("category")
df_als["product_id"] = df_als["product_id"].astype("category")
df_als["user_code"] = df_als["customer_id"].cat.codes
df_als["item_code"] = df_als["product_id"].cat.codes
als_user_map = dict(enumerate(df_als["customer_id"].cat.categories))
als_item_map = dict(enumerate(df_als["product_id"].cat.categories))
als_user_inv = {v: k for k, v in als_user_map.items()}
als_item_inv = {v: k for k, v in als_item_map.items()}

als_train_matrix = sparse.csr_matrix(
    (df_als["quantity"].astype(float), (df_als["user_code"], df_als["item_code"]))
)
als_train_scaled = als_train_matrix.multiply(40).astype("float32")

model_ALS = AlternatingLeastSquares(factors=64, regularization=0.1, iterations=20, random_state=SEED)
model_ALS.fit(als_train_scaled)
print("ALS entrenado sobre el split temporal del hibrido:", als_train_matrix.shape)


def als_score(customer_id, product_id):
    """Score crudo de ALS (producto interno usuario-item). 0 si el usuario o el producto son nuevos."""
    u = als_user_inv.get(customer_id)
    i = als_item_inv.get(product_id)
    if u is None or i is None:
        return 0.0
    return float(model_ALS.user_factors[u] @ model_ALS.item_factors[i])


print("Listo: componentes de contenido y colaborativo preparados.")

# ---------- 5. Dataset supervisado de candidatos (misma logica que la companera + feature de ALS) ----------
global_pop = train_df.groupby("product_id")["quantity"].sum().to_dict()
product_info = product_model.set_index("product_id")[["unit_price", "product_rating"]].to_dict("index")
test_user_products = test_df.groupby("customer_id")["product_id"].apply(set).to_dict()

all_train_customers = train_df["customer_id"].unique()
dataset_rows = []

print("Generando dataset supervisado (candidatos de contenido + feature de ALS)...")
t0 = time.time()
for customer_id in tqdm(all_train_customers):
    candidate_pids = evaluation_contend(customer_id, train_df, k_top=20, alpha_recompra=1.5)
    user_train_history = train_df[train_df["customer_id"] == customer_id].groupby("product_id")["quantity"].sum().to_dict()
    actual_purchases = test_user_products.get(customer_id, set())
    is_new = 1 if len(user_train_history) == 0 else 0

    positives, negatives = [], []
    for rank, prod_id in enumerate(candidate_pids):
        label = 1 if prod_id in actual_purchases else 0
        row = {
            "customer_id": customer_id,
            "product_id": prod_id,
            "content_rank_score": 1.0 / (rank + 1),
            "past_quantity": user_train_history.get(prod_id, 0),
            "global_popularity": global_pop.get(prod_id, 0),
            "unit_price": product_info.get(prod_id, {}).get("unit_price", 0),
            "product_rating": product_info.get(prod_id, {}).get("product_rating", 0),
            "is_new_customer": is_new,
            "als_score": als_score(customer_id, prod_id),
            "target": label,
        }
        (positives if label == 1 else negatives).append(row)

    dataset_rows.extend(positives)
    if positives:
        sample_size = min(len(negatives), len(positives) * 4)
        dataset_rows.extend(random.sample(negatives, sample_size))
    elif negatives:
        dataset_rows.extend(random.sample(negatives, min(len(negatives), 2)))

df_ml_full = pd.DataFrame(dataset_rows)
print(f"Dataset generado en {time.time()-t0:.1f}s: {df_ml_full.shape[0]} filas")
print(df_ml_full["target"].value_counts())
df_ml_full.to_csv("hybrid_candidates_dataset.csv", index=False)
print("Guardado en hybrid_candidates_dataset.csv")
