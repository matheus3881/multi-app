import os
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
import pyodbc
import json
import time
from sentence_transformers import SentenceTransformer
import streamlit as st

# Carrega as variáveis de ambiente a partir do .env na raiz do projeto
ENV_PATH = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=ENV_PATH, override=True)


def get_connection_string():
    """Monta a string de conexão a partir do .env / variáveis de ambiente."""
    dotenv_dict = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}

    driver = (
        os.getenv("SQL_DRIVER")
        or dotenv_dict.get("DRIVER")
        or os.getenv("DRIVER")
        or "{ODBC Driver 17 for SQL Server}"
    )
    server = (
        os.getenv("SQL_SERVER")
        or dotenv_dict.get("SERVER")
        or os.getenv("SERVER")
    )
    database = (
        os.getenv("SQL_DATABASE")
        or dotenv_dict.get("DATABASE")
        or os.getenv("DATABASE")
    )
    user = (
        os.getenv("SQL_USER")
        or dotenv_dict.get("UID")
        or os.getenv("SQL_UID")
        or os.getenv("UID")
    )
    # PWD é frequentemente sobrescrito pelo shell com o diretório atual;
    # por isso, priorizamos explicitamente o valor do .env ou SQL_PASSWORD
    password = (
        os.getenv("SQL_PASSWORD")
        or dotenv_dict.get("PWD")
        or os.getenv("PWD")
    )
    encrypt = (
        os.getenv("SQL_ENCRYPT")
        or dotenv_dict.get("Encrypt")
        or os.getenv("Encrypt")
        or "yes"
    )
    trust_cert = (
        os.getenv("SQL_TRUST_SERVER_CERTIFICATE")
        or dotenv_dict.get("TrustServerCertificate")
        or os.getenv("TrustServerCertificate")
        or "yes"
    )

    if not all([server, database, user, password]):
        missing = [
            k for k, v in [("SERVER", server), ("DATABASE", database), ("UID", user), ("PWD", password)]
            if not v
        ]
        raise ValueError(
            f"Configurações de banco de dados ausentes no .env ou variáveis de ambiente: {', '.join(missing)}"
        )

    return (
        f"DRIVER={driver};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        f"Encrypt={encrypt};"
        f"TrustServerCertificate={trust_cert};"
    )


def connect_to_sqlserver():
    """Conecta ao SQL Server utilizando credenciais carregadas com segurança do .env."""
    conn_str = get_connection_string()
    return pyodbc.connect(conn_str)


@st.cache_resource
def load_sentence_transformer():
    """Carrega o modelo de embeddings e o armazena em cache."""
    st.write("Carregando modelo 'BAAI/bge-m3' (isso acontece apenas uma vez)...")
    start_time = time.perf_counter()
    model = SentenceTransformer('BAAI/bge-m3', device='cpu')
    elapsed = time.perf_counter() - start_time
    st.success(f"Modelo carregado em {elapsed:.2f} segundos.")
    return model


def display_results(results, query_text):
    """Exibe os resultados da busca puramente vetorial no Streamlit."""
    st.divider()
    st.subheader("Resultados da Busca Vetorial")
    st.markdown(f"**Consulta**: **'{query_text}'**")
    st.markdown(f"**Total de resultados**: **{len(results)}**")

    for i, result in enumerate(results, 1):
        st.divider()
        st.markdown(f"#### Resultado {i}")
        st.markdown(f"**Similaridade Vetorial**: **{result['vector_similarity']:.4f}** (Document ID: {result['document_id']})")
        
        with st.expander("**Ver Resumo e Documento**"):
            st.markdown("**Resumo:**")
            st.write(f"{result['summary']}")
            st.markdown("**Documento (primeiros 300 caracteres):**")
            st.write(f"{result['document'][:300]}...")

    st.divider()


def search_by_sql_vector_inline(query_text, model, top_k=5):
    """Executa uma busca puramente vetorial no SQL Server."""
    
    embed_start = time.perf_counter()
    query_embedding = model.encode([query_text])[0].tolist()
    embed_time = time.perf_counter() - embed_start

    # Prepara os literais para injeção na query.
    # ATENÇÃO: Para produção, use parâmetros para evitar SQL Injection.
    query_vector_json = json.dumps(query_embedding).replace("'", "''")

    conn = connect_to_sqlserver()
    cursor = conn.cursor()

    try:
        # --- QUERY SIMPLIFICADA PARA BUSCA PURAMENTE VETORIAL ---
        sql = f"""
        SELECT TOP ({top_k})
            d.DocumentId,
            d.Document,
            d.Summary,
            (1 - VECTOR_DISTANCE('cosine', e.EmbeddingData, CAST(CAST(N'{query_vector_json}' AS NVARCHAR(MAX)) AS VECTOR(1024)))) AS VectorSimilarity
        FROM dbo.EmbeddingsFaleConosco e
        INNER JOIN dbo.DocumentsFaleConosco d ON e.DocumentId = d.DocumentId
        ORDER BY VectorSimilarity DESC;
        """

        sql_start = time.perf_counter()
        cursor.execute(sql)
        rows = cursor.fetchall()
        sql_time = time.perf_counter() - sql_start

        # Imprime os tempos no console (para debug)
        print(f"Tempos (s): embed={embed_time:.3f}, sql_exec={sql_time:.3f}")

        formatted_results = []
        for row in rows:
            formatted_results.append({
                'document_id': row.DocumentId,
                'document': row.Document,
                'summary': row.Summary,
                'vector_similarity': row.VectorSimilarity
            })

        return formatted_results

    except Exception as e:
        st.error(f"Erro na busca SQL: {e}")
        print(f"Erro na busca inline (SQL): {e}")
        return []
    
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()


def buscar_no_banco(query, top_k):
    model_object = load_sentence_transformer()
    resullts = search_by_sql_vector_inline(query_text=query, top_k=top_k, model=model_object)
    if resullts:
        display = display_results(resullts, query)
        return display
    else:
        st.info("Nenhum resultado encontrado para a sua busca.")
