# main.py
from tracemalloc import start
import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import logging
import time

# ======== CONFIG ========
api_key="XXXXXXXXXXXXXXXXXX"    ############## Reemplazar el API Key aquí ##################
client = OpenAI(api_key=api_key)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)


DF_PROD_PATH = "data/catalogo_productos.csv"
FAISS_INDEX_PATH = "data/faiss.index"
df_prod = pd.read_csv(DF_PROD_PATH)
df_prod["ventas"] = df_prod["ventas"].astype(int)
index = faiss.read_index(FAISS_INDEX_PATH)


# ========= MODELOS de salida con Pydantic =========
class ProductOut(BaseModel):
    product_id: str
    product: str
    ventas: int
    distancia: float


class RAGResponse(BaseModel):
    query: str
    products: List[ProductOut]
    explanation: str


# ========= FUNCIONES INTERNAS =========
##Funcion para convertir la petición del usuario en embbedings y así poder buscar en el índice FAISS
def embed_query(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding, dtype=np.float32).reshape(1, -1)

## Definido como un motor de recomendación  sin GPT, está basado en FAISS
def retrieve_products(query: str, k_semantic: int = 10, k_final: int = 5):
    q_emb = embed_query(query)
    distances, idxs = index.search(q_emb, k_semantic)

    candidatos = df_prod.iloc[idxs[0]].copy()
    candidatos["distancia"] = distances[0]
    candidatos = candidatos.sort_values(
        by=["ventas", "distancia"],
        ascending=[False, True]
    ).head(k_final)

    return candidatos


def rag_recommendation(query: str, k: int = 2):
    retrieved = retrieve_products(query, k_final=k)

    context = "\n\n".join(
        f"Producto: {row['product']}.\nVentas: {row['ventas']}."
        for _, row in retrieved.iterrows()
    )

    prompt = f"""
El usuario busca: "{query}"

Productos relevantes del catálogo:
{context}

Escribe una recomendación clara basada SOLO en estos productos.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un recomendador experto."},
            {"role": "user", "content": prompt}
        ]
    )

    explanation = resp.choices[0].message.content
    return retrieved, explanation


# ========= API =========

app = FastAPI(
    title="Servicio de Recomendación Semántica de Productos",
    version="1.0.0",
    description="Recomendador híbrido (embeddings + FAISS ) expuesto como API."
)



@app.get("/rag_recommend", response_model=RAGResponse)
def rag_endpoint(query: str, k: int = 2):
    start = time.perf_counter()
    retrieved, explanation = rag_recommendation(query, k)

    products = [
        ProductOut(
            product_id=row["product_id"],
            product=row["product"],
            ventas=int(row["ventas"]),
            distancia=float(row["distancia"])
        )
        for _, row in retrieved.iterrows()
    ]

    
    elapsed_ms=(time.perf_counter()-start)*1000
    logger.info( 
        f"/rag_recommend query='{query}' "
        f"k={k} "
        f"num_products={len(products)} "
        f"latency_ms={elapsed_ms:.2f}"
    )
    return RAGResponse(
        query=query,
        products=products,
        explanation=explanation
    )