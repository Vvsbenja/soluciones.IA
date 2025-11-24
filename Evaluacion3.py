import streamlit as st
import os
import json
import time
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

try:
    import psutil
except Exception:
    psutil = None

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.agents import AgentExecutor, create_openai_functions_agent 
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool, tool

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

github_token = os.getenv("GITHUB_TOKEN")
github_base_url = os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")

if github_token:
    os.environ["OPENAI_API_KEY"] = github_token
    os.environ["OPENAI_API_BASE"] = github_base_url
else:
    st.error("GITHUB_TOKEN environment variable is not set.")
    st.stop()

@tool
def escribir_archivo(nombre_archivo: str, contenido: str) -> str:
    try:
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            f.write(contenido)
        return f"Archivo '{nombre_archivo}' guardado."
    except Exception as e:
        return f"Error al guardar el archivo: {str(e)}"

st.set_page_config(page_title="RAG Evaluation & Agent", page_icon="", layout="wide")

REPORT_PDF_PATH = "/mnt/data/EP3_ISY0101_Estudiante.pdf"

def initialize_client():
    if not github_token:
        return None
    client = OpenAI(
        base_url=github_base_url,
        api_key=github_token
    )
    return client

def initialize_embeddings():
    if not github_token:
        return None
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        return embeddings
    except Exception:
        return None

def get_embeddings_langchain(embeddings_model, texts):
    try:
        if isinstance(texts[0], str):
            documents = [Document(page_content=text) for text in texts]
        else:
            documents = texts
        embeddings = embeddings_model.embed_documents([doc.page_content for doc in documents])
        return np.array(embeddings)
    except Exception:
        return None

def get_query_embedding_langchain(embeddings_model, query):
    try:
        embedding = embeddings_model.embed_query(query)
        return np.array(embedding)
    except Exception:
        return None

def evaluate_faithfulness(client, query, context, response):
    if not client:
        return 5.0
    eval_prompt = f"""Evalúa si la respuesta es fiel al contexto.

Consulta: {query}

Contexto:
{context}

Respuesta:
{response}

Responde solo un número del 1 al 10."""
    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1,
            max_tokens=10
        )
        return float(result.choices[0].message.content.strip())
    except:
        return 5.0

def evaluate_relevance(client, query, response):
    if not client:
        return 5.0
    eval_prompt = f"""Evalúa qué tan relevante es la respuesta.

Consulta: {query}

Respuesta: {response}

Responde solo un número del 1 al 10."""
    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1,
            max_tokens=10
        )
        return float(result.choices[0].message.content.strip())
    except:
        return 5.0

def evaluate_context_precision(client, query, retrieved_docs):
    if not client or not retrieved_docs:
        return 0.0
    relevant_count = 0
    for doc in retrieved_docs:
        eval_prompt = f"""¿Este documento es relevante?

Consulta: {query}

Documento: {doc['document'][:300]}...

Responde SI o NO."""
        try:
            result = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                max_tokens=5
            )
            if result.choices[0].message.content.strip().upper() == 'SI':
                relevant_count += 1
        except:
            pass
    return relevant_count / len(retrieved_docs)

def evaluate_accuracy(response, ground_truth):
    try:
        if not response or not ground_truth:
            return 0.0
        return 1.0 if response.strip().lower() == ground_truth.strip().lower() else 0.0
    except:
        return 0.0

def evaluate_consistency(query, response):
    previous = [log for log in st.session_state.interaction_logs if log["query"] == query]
    if len(previous) < 1:
        return 1.0
    last_response = previous[-1]["response"]
    return 1.0 if last_response.strip() == response.strip() else 0.0

def get_resource_usage():
    if psutil is None:
        return {"cpu": None, "ram": None}
    return {
        "cpu": psutil.cpu_percent(interval=0.1),
        "ram": psutil.virtual_memory().percent
    }

def hybrid_search_with_metrics(query, documents, embeddings, embeddings_model, client, top_k=5):
    start_time = time.time()
    query_embedding = get_query_embedding_langchain(embeddings_model, query)
    if query_embedding is None:
        return [], 0.0
    semantic_similarities = cosine_similarity([query_embedding], embeddings)[0]
    keyword_scores = []
    query_words = set(query.lower().split())
    for doc in documents:
        doc_words = set(doc.lower().split())
        overlap = len(query_words.intersection(doc_words))
        keyword_scores.append(overlap / max(len(query_words), 1))
    combined_scores = 0.7 * semantic_similarities + 0.3 * np.array(keyword_scores)
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'semantic_score': float(semantic_similarities[idx]),
            'keyword_score': float(keyword_scores[idx]),
            'combined_score': float(combined_scores[idx]),
            'index': int(idx)
        })
    retrieval_time = time.time() - start_time
    return results, retrieval_time

def generate_response_with_metrics(client, query, context_docs):
    if not client:
        return "Error", 0.0
    start_time = time.time()
    context = "".join([f"Documento {i+1}: {doc['document']}" 
                         for i, doc in enumerate(context_docs)])
    prompt = f"""Contexto:
{context}

Pregunta: {query}

Responde basándote en el contexto."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )
        generation_time = time.time() - start_time
        response_text = response.choices[0].message.content
        return response_text, generation_time
    except Exception as e:
        st.session_state.error_count = st.session_state.get('error_count', 0) + 1
        return str(e), time.time() - start_time

def create_evaluation_dataset():
    return [
        {
            "query": "¿Qué es la inteligencia artificial?",
            "expected_context": "definición de IA",
            "ground_truth": "La inteligencia artificial es una rama de la informática que busca crear máquinas capaces de realizar tareas que requieren inteligencia humana."
        },
        {
            "query": "¿Cómo funciona RAG?",
            "expected_context": "funcionamiento de RAG",
            "ground_truth": "RAG combina la búsqueda de información relevante con la generación de texto para producir respuestas más precisas."
        },
        {
            "query": "¿Qué es LangChain?",
            "expected_context": "descripción de LangChain",
            "ground_truth": "LangChain es un framework que facilita el desarrollo de aplicaciones con modelos de lenguaje."
        }
    ]

def log_interaction(query, response, metrics, context_docs):
    if 'interaction_logs' not in st.session_state:
        st.session_state.interaction_logs = []
    resources = get_resource_usage()
    consistency = evaluate_consistency(query, response)
    log_entry = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'response': response,
        'metrics': metrics,
        'context_count': len(context_docs),
        'context_scores': [doc.get('combined_score', 0) for doc in context_docs],
        'resources': resources,
        'consistency': consistency
    }
    st.session_state.interaction_logs.append(log_entry)

def export_langsmith_format(logs):
    langsmith_data = []
    for log in logs:
        langsmith_data.append({
            "run_id": log['id'],
            "timestamp": log['timestamp'],
            "inputs": {"query": log['query']},
            "outputs": {"response": log['response']},
            "metrics": log['metrics'],
            "metadata": {
                "context_count": log['context_count'],
                "context_scores": log['context_scores']
            }
        })
    return langsmith_data

@st.cache_resource
def get_agent_executor(_client, _embeddings_model):
    def run_rag_query(query: str) -> str:
        if 'eval_rag' not in st.session_state or \
           st.session_state.eval_rag.get('embeddings') is None or \
           st.session_state.eval_rag.get('embeddings_model') is None:
            return "Error: embeddings no generados."
        docs, _ = hybrid_search_with_metrics(
            query,
            st.session_state.eval_rag['documents'],
            st.session_state.eval_rag['embeddings'],
            _embeddings_model,
            _client,
            top_k=3
        )
        if not docs:
            return "No hay datos."
        response_text, _ = generate_response_with_metrics(_client, query, docs)
        return response_text

    rag_tool = Tool(
        name="consulta_documentos_huertohogar",
        func=run_rag_query,
        description="Busca información en documentos internos."
    )

    web_search_tool = DuckDuckGoSearchRun(
        name="busqueda_web",
        description="Busca información en internet."
    )

    tools = [rag_tool, escribir_archivo, web_search_tool]

    llm = ChatOpenAI(
        model="gpt-4o",
        base_url=_client.base_url,
        api_key=_client.api_key,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente con herramientas."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True
    )
    
    return agent_executor

def detect_bottlenecks(metrics):
    issues = []
    if metrics.get("retrieval_time", 0) > 1.0:
        issues.append("Retrieval lento")
    if metrics.get("generation_time", 0) > 2.0:
        issues.append("Generación lenta")
    if metrics.get("docs_retrieved", 0) > 6:
        issues.append("Demasiados documentos")
    if metrics.get("avg_relevance_score", 0) < 0.2:
        issues.append("Baja relevancia")
    return issues

def detect_anomalies(logs):
    if not logs:
        return []
    latencies = [log["metrics"]["total_time"] for log in logs if "metrics" in log and "total_time" in log["metrics"]]
    if len(latencies) < 2:
        return []
    avg = np.mean(latencies)
    sd = np.std(latencies)
    anomalies = []
    for log in logs:
        t = log["metrics"].get("total_time", None)
        if t is not None and t > avg + 2 * sd:
            anomalies.append(log)
    return anomalies

def generate_recommendations():
    logs = st.session_state.interaction_logs if 'interaction_logs' in st.session_state else []
    recs = []
    if not logs:
        recs.append("No hay logs.")
        return recs
    anomalies = detect_anomalies(logs)
    if anomalies:
        recs.append("Hay latencias anómalas.")
    if st.session_state.get('error_count', 0) > 0:
        recs.append("Hay errores registrados.")
    high_cpu = [l for l in logs if l.get('resources', {}).get('cpu') and l['resources']['cpu'] > 80]
    if high_cpu:
        recs.append("Picos de CPU.")
    inconsistencies = [l for l in logs if l.get('consistency') == 0.0]
    if inconsistencies:
        recs.append("Hay inconsistencias.")
    recs.append("Revisar top_k.")
    return recs

def main():
    st.title("Agente y RAG Observabilidad")

    if not github_token:
        st.error("Error de token.")
        return
    
    if "eval_rag" not in st.session_state:
        st.session_state.eval_rag = {
            'documents': [
                "HuertoHogar es una tienda online chilena.",
                "Los modelos de lenguaje permiten asistentes.",
                "RAG combina búsqueda con generación.",
                "LangChain facilita pipelines.",
                "Prompt engineering en HuertoHogar.",
                "Embeddings representan descripciones.",
                "La búsqueda semántica usa embeddings.",
                "Sistemas de evaluación miden precisión."
            ],
            'embeddings': None,
            'embeddings_model': None,
            'enable_logging': True
        }
    
    if 'interaction_logs' not in st.session_state:
        st.session_state.interaction_logs = []
    if 'agent_chat_history' not in st.session_state:
        st.session_state.agent_chat_history = []
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0

    client = initialize_client()
    if not client:
        return
    
    if st.session_state.eval_rag['embeddings_model'] is None:
        try:
            st.session_state.eval_rag['embeddings_model'] = initialize_embeddings()
        except:
            pass

    tab_agente, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Agente", 
        "Consulta RAG", 
        "Documentos", 
        "Métricas", 
        "Evaluación", 
        "Analytics"
    ])
    
    with tab_agente:
        st.header("Agente funcional")
        
        if st.session_state.eval_rag['embeddings'] is None:
            st.error("Embeddings no generados.")
            st.stop() 

        agent_executor = get_agent_executor(
            client, 
            st.session_state.eval_rag['embeddings_model']
        )

        for message in st.session_state.agent_chat_history:
            with st.chat_message(message.type):
                st.markdown(message.content)

        if prompt := st.chat_input("Pregunta al agente"):
            st.session_state.agent_chat_history.append(HumanMessage(content=prompt))
            with st.chat_message("human"):
                st.markdown(prompt)
            with st.chat_message("ai"):
                try:
                    response = agent_executor.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.agent_chat_history
                    })
                    response_text = response['output']
                    st.session_state.agent_chat_history.append(AIMessage(content=response_text))
                    st.markdown(response_text)
                except Exception as e:
                    st.error(str(e))
                    st.session_state.agent_chat_history.pop()

    with tab1:
        st.header("Consulta RAG")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("Pregunta:")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                top_k = st.slider("Docs:", 1, 8, 3)
            with col_b:
                eval_enabled = st.checkbox("Evaluación", value=True)
            with col_c:
                st.session_state.eval_rag['enable_logging'] = st.checkbox("Logging", value=True)
        
        with col2:
            if st.button("Generar Embeddings"):
                if st.session_state.eval_rag['documents'] and st.session_state.eval_rag['embeddings_model']:
                    embeddings = get_embeddings_langchain(
                        st.session_state.eval_rag['embeddings_model'],
                        st.session_state.eval_rag['documents']
                    )
                    if embeddings is not None:
                        st.session_state.eval_rag['embeddings'] = embeddings
                        st.success("Embeddings listos.")
                    else:
                        st.error("Error generando embeddings")

        if st.button("Consultar") and query:
            if st.session_state.eval_rag['embeddings'] is None:
                st.warning("Genera embeddings.")
            else:
                results, retrieval_time = hybrid_search_with_metrics(
                    query, 
                    st.session_state.eval_rag['documents'],
                    st.session_state.eval_rag['embeddings'],
                    st.session_state.eval_rag['embeddings_model'],
                    client,
                    top_k
                )
                if not results:
                    st.error("Error en búsqueda")
                    return
                response, generation_time = generate_response_with_metrics(client, query, results)
                metrics = {
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                    'total_time': retrieval_time + generation_time,
                    'docs_retrieved': len(results),
                    'avg_relevance_score': np.mean([r['combined_score'] for r in results])
                }
                issues = detect_bottlenecks(metrics)
                if eval_enabled:
                    context_text = "".join([r['document'] for r in results])
                    metrics['faithfulness'] = evaluate_faithfulness(client, query, context_text, response)
                    metrics['relevance'] = evaluate_relevance(client, query, response)
                    metrics['context_precision'] = evaluate_context_precision(client, query, results)
                    eval_dataset = create_evaluation_dataset()
                    matched = next((e for e in eval_dataset if e['query'].strip().lower() == query.strip().lower()), None)
                    if matched:
                        metrics['accuracy'] = evaluate_accuracy(response, matched['ground_truth'])
                    else:
                        metrics['accuracy'] = None
                    metrics['consistency'] = evaluate_consistency(query, response)
                
                st.subheader("Documentos recuperados")
                for i, result in enumerate(results):
                    with st.expander(f"Doc {i+1} - Score: {result['combined_score']:.3f}"):
                        st.write(result['document'])
                
                st.subheader("Respuesta")
                st.write(response)
                
                st.subheader("Métricas")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tiempo total", f"{metrics['total_time']:.2f}s")
                with col2:
                    st.metric("Recuperación", f"{metrics['retrieval_time']:.2f}s")
                with col3:
                    st.metric("Generación", f"{metrics['generation_time']:.2f}s")
                with col4:
                    st.metric("Docs", metrics['docs_retrieved'])
                
                if eval_enabled:
                    st.subheader("Calidad")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fidelidad", f"{metrics.get('faithfulness', 0):.1f}")
                    with col2:
                        st.metric("Relevancia", f"{metrics.get('relevance', 0):.1f}")
                    with col3:
                        st.metric("Precisión", f"{metrics.get('context_precision', 0):.2f}")
                    col4, col5 = st.columns(2)
                    with col4:
                        acc = metrics.get('accuracy', None)
                        if acc is not None:
                            st.metric("Accuracy", f"{acc:.2f}")
                        else:
                            st.write("Accuracy: N/A")
                    with col5:
                        st.metric("Consistencia", f"{metrics.get('consistency', 1.0):.2f}")
                
                if issues:
                    st.warning(", ".join(issues))
                
                if st.session_state.eval_rag['enable_logging']:
                    log_interaction(query, response, metrics, results)

    with tab2:
        st.header("Documentos")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Documentos actuales")
            for i, doc in enumerate(st.session_state.eval_rag['documents']):
                with st.expander(f"Documento {i+1} ({len(doc)} chars)"):
                    st.text_area(
                        f"Contenido D{i+1}",
                        value=doc,
                        height=100,
                        key=f"doc_display_{i}",
                        disabled=True
                    )
                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(f"Editar {i}", key=f"edit_{i}"):
                            st.session_state[f'editing_doc_{i}'] = True
                    with col_delete:
                        if st.button(f"Eliminar {i}", key=f"delete_{i}"):
                            st.session_state.eval_rag['documents'].pop(i)
                            st.session_state.eval_rag['embeddings'] = None
                            st.rerun()
                    if st.session_state.get(f'editing_doc_{i}', False):
                        new_content = st.text_area(
                            "Editar",
                            value=doc,
                            height=150,
                            key=f"edit_content_{i}"
                        )
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button(f"Guardar {i}", key=f"save_{i}"):
                                st.session_state.eval_rag['documents'][i] = new_content
                                st.session_state[f'editing_doc_{i}'] = False
                                st.session_state.eval_rag['embeddings'] = None
                                st.success("Actualizado")
                                st.rerun()
                        with col_cancel:
                            if st.button(f"Cancelar {i}", key=f"cancel_{i}"):
                                st.session_state[f'editing_doc_{i}'] = False
                                st.rerun()
        
        with col2:
            st.subheader("Agregar documento")
            new_doc = st.text_area(
                "Contenido:",
                height=200
            )
            if st.button("Agregar"):
                if new_doc.strip():
                    st.session_state.eval_rag['documents'].append(new_doc.strip())
                    st.session_state.eval_rag['embeddings'] = None
                    st.success("Agregado")
                    st.rerun()
                else:
                    st.warning("Documento vacío")
            
            st.subheader("Estadísticas")
            st.metric("Total", len(st.session_state.eval_rag['documents']))
            if st.session_state.eval_rag['documents']:
                avg_length = np.mean([len(doc) for doc in st.session_state.eval_rag['documents']])
                st.metric("Promedio chars", f"{avg_length:.0f}")
                total_words = sum([len(doc.split()) for doc in st.session_state.eval_rag['documents']])
                st.metric("Palabras", f"{total_words}")
            
            st.subheader("Acciones")
            
            if st.button("Limpiar todo"):
                st.session_state.eval_rag['documents'] = []
                st.session_state.eval_rag['embeddings'] = None
                st.success("Eliminado")
                st.rerun()
            
            uploaded_file = st.file_uploader(
                "Cargar archivo",
                type=['txt', 'md']
            )
            
            if uploaded_file is not None:
                try:
                    content = uploaded_file.read().decode('utf-8')
                    if st.button("Importar archivo"):
                        st.session_state.eval_rag['documents'].append(content)
                        st.session_state.eval_rag['embeddings'] = None
                        st.success("Importado")
                        st.rerun()
                except Exception as e:
                    st.error(str(e))
            
            st.subheader("Estado")
            if st.session_state.eval_rag['embeddings'] is not None:
                st.success("Embeddings listos")
            else:
                st.warning("Sin embeddings")

    with tab3:
        st.header("Dashboard")
        
        if st.session_state.interaction_logs:
            df_rows = []
            for log in st.session_state.interaction_logs:
                row = {
                    'timestamp': log['timestamp'],
                    'query': log['query'],
                    'response_length': len(log['response']) if log['response'] else 0,
                    'total_time': log['metrics'].get('total_time', None) if 'metrics' in log else None,
                    'retrieval_time': log['metrics'].get('retrieval_time', None),
                    'generation_time': log['metrics'].get('generation_time', None),
                    'docs_retrieved': log['metrics'].get('docs_retrieved', None),
                    'avg_relevance_score': log['metrics'].get('avg_relevance_score', None),
                    'faithfulness': log['metrics'].get('faithfulness', None),
                    'relevance': log['metrics'].get('relevance', None),
                    'accuracy': log['metrics'].get('accuracy', None),
                    'consistency': log.get('consistency', None),
                    'cpu': log.get('resources', {}).get('cpu', None),
                    'ram': log.get('resources', {}).get('ram', None)
                }
                df_rows.append(row)
            df = pd.DataFrame(df_rows)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Tiempos")
                fig = px.line(df, x='timestamp', y='total_time')
                st.plotly_chart(fig, use_container_width=True)
                
                if 'faithfulness' in df.columns:
                    st.subheader("Fidelidad")
                    fig = px.histogram(df, x='faithfulness')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Retrieval vs Gen")
                fig = px.scatter(df, x='retrieval_time', y='generation_time', size='docs_retrieved')
                st.plotly_chart(fig, use_container_width=True)
                
                if 'relevance' in df.columns and 'consistency' in df.columns:
                    st.subheader("Relevancia vs Consistencia")
                    fig = px.scatter(df, x='relevance', y='consistency')
                    st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("CPU / RAM")
            if df['cpu'].notnull().any():
                fig = px.line(df, x='timestamp', y='cpu')
                st.plotly_chart(fig, use_container_width=True)
            if df['ram'].notnull().any():
                fig = px.line(df, x='timestamp', y='ram')
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Accuracy y Consistencia")
            if df['accuracy'].notnull().any():
                avg_acc = df['accuracy'].dropna().mean()
                st.metric("Accuracy", f"{avg_acc:.2f}")
                fig = px.histogram(df, x='accuracy')
                st.plotly_chart(fig, use_container_width=True)
            if 'consistency' in df.columns:
                avg_cons = df['consistency'].dropna().mean()
                st.metric("Consistencia", f"{avg_cons:.2f}")
            
            st.subheader("Errores")
            st.metric("Errores", st.session_state.get('error_count', 0))
            
            anomalies = detect_anomalies(st.session_state.interaction_logs)
            st.subheader("Anomalías")
            st.write(len(anomalies))
            for a in anomalies:
                st.write(a['id'], a['timestamp'], a['metrics'].get('total_time'))
            
            st.subheader("Recomendaciones")
            recs = generate_recommendations()
            for r in recs:
                st.write("- " + r)
        else:
            st.info("No hay interacciones.")

    with tab4:
        st.header("Evaluación")
        eval_dataset = create_evaluation_dataset()
        if st.button("Ejecutar evaluación"):
            results = []
            for item in eval_dataset:
                q = item['query']
                docs, rt = hybrid_search_with_metrics(
                    q,
                    st.session_state.eval_rag['documents'],
                    st.session_state.eval_rag['embeddings'],
                    st.session_state.eval_rag['embeddings_model'],
                    client,
                    top_k=3
                )
                resp, gen_time = generate_response_with_metrics(client, q, docs)
                acc = evaluate_accuracy(resp, item['ground_truth'])
                consistency = evaluate_consistency(q, resp)
                metrics = {
                    'retrieval_time': rt,
                    'generation_time': gen_time,
                    'total_time': rt + gen_time,
                    'docs_retrieved': len(docs),
                    'avg_relevance_score': np.mean([d['combined_score'] for d in docs]) if docs else 0,
                    'accuracy': acc
                }
                log_interaction(q, resp, metrics, docs)
                results.append({
                    'query': q,
                    'response': resp,
                    'accuracy': acc
                })
            df_eval = pd.DataFrame(results)
            st.subheader("Resultados")
            st.dataframe(df_eval)
            st.success("Evaluación lista.")

    with tab5:
        st.header("Analytics / Export")
        if st.session_state.interaction_logs:
            if st.button("Exportar logs JSON"):
                data = json.dumps(st.session_state.interaction_logs, indent=2, ensure_ascii=False)
                escribir_archivo("interaction_logs_ep3.json", data)
                st.success("Exportado.")
            if st.button("Generar informe"):
                recs = generate_recommendations()
                draft = [
                    "Informe EP3",
                    "Fecha: " + datetime.now().isoformat(),
                    "Métricas clave:",
                    f"Interacciones: {len(st.session_state.interaction_logs)}",
                    f"Errores: {st.session_state.get('error_count', 0)}",
                    "",
                    "Recomendaciones:"
                ]
                draft.extend([f"- {r}" for r in recs])
                content = "\n".join(draft)
                escribir_archivo("informe_ep3_borrador.txt", content)
                st.success("Informe creado.")
        else:
            st.info("No hay logs.")

if __name__ == "__main__":
    main()
