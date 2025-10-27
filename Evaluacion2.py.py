# File: ia.py
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
    st.warning("⚠️ python-dotenv no está instalado. Instálalo con: pip install python-dotenv")


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("⚠️ python-dotenv no está instalado. Instálalo con: pip install python-dotenv")

github_token = os.getenv("GITHUB_TOKEN")
github_base_url = os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")

if github_token:
    os.environ["OPENAI_API_KEY"] = github_token
    os.environ["OPENAI_API_BASE"] = github_base_url
else:
    st.error("❌ GITHUB_TOKEN environment variable is not set. Please check your .env file.")
    st.info("💡 Make sure your .env file contains: GITHUB_TOKEN=your_token_here")
    st.stop()


@tool
def escribir_archivo(nombre_archivo: str, contenido: str) -> str:
    """
    Escribe el 'contenido' de texto en un archivo con el 'nombre_archivo' especificado.
    Útil para guardar resúmenes o resultados.
    """
    try:
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            f.write(contenido)
        return f"Archivo '{nombre_archivo}' guardado exitosamente."
    except Exception as e:
        return f"Error al guardar el archivo: {str(e)}"


st.set_page_config(page_title="RAG Evaluation & Agent", page_icon="🤖", layout="wide")

def initialize_client():
    if not github_token:
        st.error("❌ GitHub token not available")
        return None
    
    client = OpenAI(
        base_url=github_base_url,
        api_key=github_token
    )
    return client

def initialize_embeddings():
    """Initialize LangChain embeddings model"""
    if not github_token:
        st.error("❌ GitHub token not available for embeddings")
        return None
    
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

def get_embeddings_langchain(embeddings_model, texts):
    """Get embeddings using LangChain"""
    try:
        if isinstance(texts[0], str):
            documents = [Document(page_content=text) for text in texts]
        else:
            documents = texts
        
        embeddings = embeddings_model.embed_documents([doc.page_content for doc in documents])
        return np.array(embeddings)
    except Exception as e:
        st.error(f"Error getting embeddings: {str(e)}")
        return None

def get_query_embedding_langchain(embeddings_model, query):
    """Get query embedding using LangChain"""
    try:
        embedding = embeddings_model.embed_query(query)
        return np.array(embedding)
    except Exception as e:
        st.error(f"Error getting query embedding: {str(e)}")
        return None

def evaluate_faithfulness(client, query, context, response):
    if not client:
        return 5.0
        
    eval_prompt = f"""Evalúa si la respuesta es fiel al contexto proporcionado.

Consulta: {query}

Contexto:
{context}

Respuesta:
{response}

¿La respuesta está basada únicamente en la información del contexto? 
Responde con un número del 1-10 donde:
- 1-3: Respuesta contradice o no está basada en el contexto
- 4-6: Respuesta parcialmente basada en el contexto
- 7-10: Respuesta completamente fiel al contexto

Responde SOLO con el número:"""

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
        
    eval_prompt = f"""Evalúa qué tan relevante es la respuesta para la consulta.

Consulta: {query}

Respuesta: {response}

¿Qué tan bien responde la respuesta a la consulta?
Responde con un número del 1-10 donde:
- 1-3: Respuesta no relacionada o irrelevante
- 4-6: Respuesta parcialmente relevante
- 7-10: Respuesta muy relevante y útil

Responde SOLO con el número:"""

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
        eval_prompt = f"""¿Este documento es relevante para responder la consulta?

Consulta: {query}

Documento: {doc['document'][:300]}...

Responde SOLO 'SI' o 'NO':"""
        
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
            'semantic_score': semantic_similarities[idx],
            'keyword_score': keyword_scores[idx],
            'combined_score': combined_scores[idx],
            'index': idx
        })
    
    retrieval_time = time.time() - start_time
    
    return results, retrieval_time

def generate_response_with_metrics(client, query, context_docs):
    if not client:
        return "Error: Cliente no disponible", 0.0
        
    start_time = time.time()
    
    context = "".join([f"Documento {i+1}: {doc['document']}" 
                         for i, doc in enumerate(context_docs)])
    
    prompt = f"""Contexto:
{context}

Pregunta: {query}

Responde basándote únicamente en el contexto proporcionado."""

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
        return f"Error generating response: {str(e)}", time.time() - start_time

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
    
    log_entry = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'response': response,
        'metrics': metrics,
        'context_count': len(context_docs),
        'context_scores': [doc.get('combined_score', 0) for doc in context_docs]
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


### --- MODIFICADO: Fábrica de Agente (con importaciones v1.0) --- ###
@st.cache_resource
def get_agent_executor(_client, _embeddings_model):
    """
    Crea y devuelve el AgentExecutor.
    Define las herramientas que el agente puede usar.
    """
    
    # 1. Herramienta RAG (Consulta de Documentos Internos)
    def run_rag_query(query: str) -> str:
        """
        Responde preguntas basándose *únicamente* en los documentos internos 
        cargados sobre HuertoHogar. Útil para preguntas sobre el catálogo, 
        políticas internas o contenido del blog.
        """
        if 'eval_rag' not in st.session_state or \
           st.session_state.eval_rag.get('embeddings') is None or \
           st.session_state.eval_rag.get('embeddings_model') is None:
            return "Error: Los embeddings RAG no han sido generados. Ve a la pestaña 'Consulta RAG' y genera los embeddings primero."

        docs, _ = hybrid_search_with_metrics(
            query,
            st.session_state.eval_rag['documents'],
            st.session_state.eval_rag['embeddings'],
            _embeddings_model,
            _client,
            top_k=3
        )
        
        if not docs:
            return "No se encontró información relevante en los documentos internos."
            
        response_text, _ = generate_response_with_metrics(_client, query, docs)
        return response_text

    ### CAMBIO v1.0: Tool ahora se importa de langchain_core.tools ###
    rag_tool = Tool(
        name="consulta_documentos_huertohogar",
        func=run_rag_query,
        description="Busca y responde preguntas usando *exclusivamente* los documentos internos de HuertoHogar (catálogo, blogs, políticas). No usar para información general o de internet."
    )

    # 2. Herramienta de Búsqueda Web (Consulta Externa)
    web_search_tool = DuckDuckGoSearchRun(
        name="busqueda_web",
        description="Busca información actualizada en internet usando DuckDuckGo. Útil para eventos actuales, competidores, precios (ej. dólar) o información general que no está en los documentos de HuertoHogar."
    )

    # 3. Lista de Herramientas
    # (La herramienta 'escribir_archivo' ya se definió globalmente con @tool)
    tools = [rag_tool, escribir_archivo, web_search_tool]

    # 4. LLM y Prompt del Agente
    llm = ChatOpenAI(
        model="gpt-4o",
        base_url=_client.base_url,
        api_key=_client.api_key,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente de IA experto. Tu trabajo es ayudar al usuario. Tienes varias herramientas a tu disposición. Debes razonar paso a paso qué herramienta usar. Si la información no está en los documentos de HuertoHogar, usa la búsqueda web. Si te piden guardar algo, usa la herramienta para escribir archivos."),
        MessagesPlaceholder(variable_name="chat_history"), # Memoria (IL2.2)
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad") # Lugar donde el agente "piensa"
    ])

    # 5. Crear el Agente y el Ejecutor
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True # Muestra el razonamiento en la consola
    )
    
    return agent_executor
### --- Fin Fábrica de Agente --- ###


def main():
    st.title("🤖 Agente Funcional y Evaluación RAG (ISY0101)")
    
    if not github_token:
        st.error("❌ Please check your .env file and make sure GITHUB_TOKEN is set.")
        st.info("💡 Your .env file should contain: GITHUB_TOKEN=your_token_here")
        return
    
    if "eval_rag" not in st.session_state:
        st.session_state.eval_rag = {
            'documents': [
                "HuertoHogar es una tienda online chilena que conecta a consumidores con agricultores locales, entregando productos frescos (frutas, verduras, lácteos y productos orgánicos) directamente a la puerta del cliente.",
                "Los modelos de lenguaje grande (LLM) permiten crear asistentes que entienden consultas de clientes sobre disponibilidad, origen de productos, recetas y recomendaciones personalizadas basadas en el historial de compras de HuertoHogar.",
                "RAG (Retrieval-Augmented Generation) combina la búsqueda de fragmentos relevantes del catálogo y fichas de producto de HuertoHogar con la generación de texto del LLM para producir respuestas precisas y verificables sobre los productos y servicios.",
                "LangChain es un framework que facilita el desarrollo del pipeline RAG y de agentes (ingesta, embeddings, retrieval y chains) para casos como HuertoHogar, simplificando la integración entre el catálogo, la base de datos y el modelo de lenguaje.",
                "Prompt engineering en el contexto de HuertoHogar es la práctica de diseñar instrucciones claras para que el asistente devuelva: disponibilidad exacta, origen del producto, recetas paso a paso o recomendaciones acorde a preferencias del cliente.",
                "Los embeddings representan descripciones de producto, recetas, reseñas y contenido del blog de HuertoHogar como vectores en un espacio semántico, permitiendo medir similitud por significado y no sólo por coincidencia de palabras.",
                "La búsqueda semántica usa embeddings para encontrar productos o contenidos relacionados por significado (por ejemplo: buscar 'snack saludable' retorna frutas y mezclas de frutos secos aunque no aparezca la frase exacta en el catálogo).",
                "Los sistemas de evaluación para HuertoHogar miden métricas como relevancia (¿la respuesta atiende la consulta?), fidelidad (¿la respuesta está basada en el catálogo?) y precisión del contexto (proporción de documentos recuperados realmente útiles)."
            ],
            'embeddings': None,
            'embeddings_model': None,
            'enable_logging': True
        }
    
    if 'interaction_logs' not in st.session_state:
        st.session_state.interaction_logs = []
    
    if 'agent_chat_history' not in st.session_state:
        st.session_state.agent_chat_history = []
        
    client = initialize_client()
    if not client:
        st.error("❌ Failed to initialize OpenAI client")
        return
    
    if st.session_state.eval_rag['embeddings_model'] is None:
        try:
            st.session_state.eval_rag['embeddings_model'] = initialize_embeddings()
        except Exception as e:
            st.error(f"Error inicializando embeddings: {str(e)}")

    tab_agente, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Agente Funcional", 
        "🔍 Consulta RAG", 
        "📄 Documentos", 
        "📊 Métricas", 
        "🧪 Evaluación", 
        "📈 Analytics"
    ])
    
    with tab_agente:
        st.header("🤖 Agente Funcional con Planificación y Herramientas")
        st.info(
            "Este es el agente que cumple con la Evaluación Parcial 2 (IL2.1, IL2.2, IL2.3).\n"
            "Puede **planificar** tareas, **recordar** la conversación (memoria) y **usar herramientas**:\n"
            "1.  **Búsqueda Web (DuckDuckGo)**: Para consultas de internet (ej. '¿cuál es el precio del dólar?').\n"
            "2.  **Consulta RAG**: Para consultar los documentos internos (ej. '¿Qué es HuertoHogar?').\n"
            "3.  **Escribir Archivos**: Para guardar resultados (ej. 'guarda un resumen en resumen.txt')."
        )
        
        if st.session_state.eval_rag['embeddings'] is None:
            st.error("⚠️ **Acción Requerida:** Los embeddings RAG no están generados.")
            st.warning("Por favor, ve a la pestaña '🔍 Consulta RAG' y presiona el botón '🔄 Generar Embeddings (LangChain)' para activar la herramienta de consulta de documentos.")
            st.stop() 

        agent_executor = get_agent_executor(
            client, 
            st.session_state.eval_rag['embeddings_model']
        )

        for message in st.session_state.agent_chat_history:
            with st.chat_message(message.type):
                st.markdown(message.content)

        if prompt := st.chat_input("Pregúntale al agente (ej. '¿Qué es RAG según los documentos? y luego busca el clima actual en Chile')"):
            st.session_state.agent_chat_history.append(HumanMessage(content=prompt))
            with st.chat_message("human"):
                st.markdown(prompt)

            with st.chat_message("ai"):
                with st.spinner("Agente pensando y planificando..."):
                    try:
                        response = agent_executor.invoke({
                            "input": prompt,
                            "chat_history": st.session_state.agent_chat_history
                        })
                        
                        response_text = response['output']
                        st.session_state.agent_chat_history.append(AIMessage(content=response_text))
                        st.markdown(response_text)
                        
                        if "escribir_archivo" in str(response.get("intermediate_steps", "")):
                            st.success("¡El agente usó la herramienta `escribir_archivo`! Revisa tu carpeta local.")
                            
                    except Exception as e:
                        st.error(f"Error al procesar la solicitud del agente: {e}")
                        st.session_state.agent_chat_history.pop()


    with tab1:
        st.header("💬 Consulta RAG con Monitoreo (Tu código original)")
        st.info("Esta pestaña es tu sistema RAG original, usado para evaluación detallada y como herramienta para el agente.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("Haz tu pregunta (solo RAG):")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                top_k = st.slider("Docs a recuperar:", 1, 8, 3)
            with col_b:
                eval_enabled = st.checkbox("Evaluación automática", value=True)
            with col_c:
                st.session_state.eval_rag['enable_logging'] = st.checkbox("Logging", value=True)
        
        with col2:
            st.write("") 
            st.write("") 
            if st.button("🔄 Generar Embeddings (LangChain)"):
                if st.session_state.eval_rag['documents'] and st.session_state.eval_rag['embeddings_model']:
                    with st.spinner("Generando embeddings con LangChain..."):
                        embeddings = get_embeddings_langchain(
                            st.session_state.eval_rag['embeddings_model'],
                            st.session_state.eval_rag['documents']
                        )
                        if embeddings is not None:
                            st.session_state.eval_rag['embeddings'] = embeddings
                            st.success("✅ Embeddings listos con LangChain")
                            st.info("¡Ahora puedes usar el '🤖 Agente Funcional'!")
                        else:
                            st.error("❌ Error generando embeddings")
                else:
                    st.warning("Modelo de embeddings no disponible")
        
        if st.button("🚀 Consultar RAG con Métricas") and query:
            if st.session_state.eval_rag['embeddings'] is None:
                st.warning("Genera embeddings primero")
            elif st.session_state.eval_rag['embeddings_model'] is None:
                st.warning("Modelo de embeddings no inicializado")
            else:
                with st.spinner("Procesando con métricas..."):
                    results, retrieval_time = hybrid_search_with_metrics(
                        query, 
                        st.session_state.eval_rag['documents'],
                        st.session_state.eval_rag['embeddings'],
                        st.session_state.eval_rag['embeddings_model'],
                        client,
                        top_k
                    )
                    
                    if not results:
                        st.error("Error en la búsqueda")
                        return
                    
                    response, generation_time = generate_response_with_metrics(client, query, results)
                    
                    metrics = {
                        'retrieval_time': retrieval_time,
                        'generation_time': generation_time,
                        'total_time': retrieval_time + generation_time,
                        'docs_retrieved': len(results),
                        'avg_relevance_score': np.mean([r['combined_score'] for r in results])
                    }
                    
                    if eval_enabled:
                        context_text = "".join([r['document'] for r in results])
                        
                        with st.spinner("Evaluando calidad..."):
                            metrics['faithfulness'] = evaluate_faithfulness(client, query, context_text, response)
                            metrics['relevance'] = evaluate_relevance(client, query, response)
                            metrics['context_precision'] = evaluate_context_precision(client, query, results)
                    
                    st.subheader("📋 Documentos Recuperados")
                    for i, result in enumerate(results):
                        with st.expander(f"Doc {i+1} - Score: {result['combined_score']:.3f}"):
                            st.write(result['document'])
                    
                    st.subheader("🤖 Respuesta")
                    st.write(response)
                    
                    st.subheader("⏱️ Métricas de Rendimiento")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Tiempo total", f"{metrics['total_time']:.2f}s")
                    with col2:
                        st.metric("Recuperación", f"{metrics['retrieval_time']:.2f}s")
                    with col3:
                        st.metric("Generación", f"{metrics['generation_time']:.2f}s")
                    with col4:
                        st.metric("Docs recuperados", metrics['docs_retrieved'])
                    
                    if eval_enabled:
                        st.subheader("🎯 Métricas de Calidad")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Fidelidad", f"{metrics['faithfulness']:.1f}/10")
                        with col2:
                            st.metric("Relevancia", f"{metrics['relevance']:.1f}/10")
                        with col3:
                            st.metric("Precisión contexto", f"{metrics['context_precision']:.2f}")
                    
                    if st.session_state.eval_rag['enable_logging']:
                        log_interaction(query, response, metrics, results)

    # --- El resto de tus pestañas (tab2, tab3, tab4, tab5) van aquí sin cambios ---
    with tab2:
        st.header("📄 Gestión de Documentos")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📚 Documentos Actuales")
            
            for i, doc in enumerate(st.session_state.eval_rag['documents']):
                with st.expander(f"Documento {i+1} ({len(doc)} caracteres)"):
                    st.text_area(
                        f"Contenido del documento {i+1}:",
                        value=doc,
                        height=100,
                        key=f"doc_display_{i}",
                        disabled=True
                    )
                    
                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(f"✏️ Editar", key=f"edit_{i}"):
                            st.session_state[f'editing_doc_{i}'] = True
                    
                    with col_delete:
                        if st.button(f"🗑️ Eliminar", key=f"delete_{i}"):
                            st.session_state.eval_rag['documents'].pop(i)
                            st.session_state.eval_rag['embeddings'] = None
                            st.rerun()
                    
                    if st.session_state.get(f'editing_doc_{i}', False):
                        new_content = st.text_area(
                            "Editar contenido:",
                            value=doc,
                            height=150,
                            key=f"edit_content_{i}"
                        )
                        
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button(f"💾 Guardar", key=f"save_{i}"):
                                st.session_state.eval_rag['documents'][i] = new_content
                                st.session_state[f'editing_doc_{i}'] = False
                                st.session_state.eval_rag['embeddings'] = None
                                st.success("Documento actualizado")
                                st.rerun()
                        
                        with col_cancel:
                            if st.button(f"❌ Cancelar", key=f"cancel_{i}"):
                                st.session_state[f'editing_doc_{i}'] = False
                                st.rerun()
        
        with col2:
            st.subheader("➕ Agregar Documento")
            
            new_doc = st.text_area(
                "Contenido del nuevo documento:",
                height=200,
                placeholder="Escribe aquí el contenido del nuevo documento..."
            )
            
            if st.button("📝 Agregar Documento"):
                if new_doc.strip():
                    st.session_state.eval_rag['documents'].append(new_doc.strip())
                    st.session_state.eval_rag['embeddings'] = None
                    st.success("Documento agregado exitosamente")
                    st.rerun()
                else:
                    st.warning("El documento no puede estar vacío")
            
            st.subheader("📊 Estadísticas")
            st.metric("Total documentos", len(st.session_state.eval_rag['documents']))
            
            if st.session_state.eval_rag['documents']:
                avg_length = np.mean([len(doc) for doc in st.session_state.eval_rag['documents']])
                st.metric("Longitud promedio", f"{avg_length:.0f} caracteres")
                
                total_words = sum([len(doc.split()) for doc in st.session_state.eval_rag['documents']])
                st.metric("Total palabras", f"{total_words:,}")
            
            st.subheader("🔄 Acciones")
            
            if st.button("🗑️ Limpiar Todos"):
                if st.session_state.eval_rag['documents']:
                    st.session_state.eval_rag['documents'] = []
                    st.session_state.eval_rag['embeddings'] = None
                    st.success("Todos los documentos eliminados")
                    st.rerun()
            
            uploaded_file = st.file_uploader(
                "📁 Cargar archivo de texto",
                type=['txt', 'md'],
                help="Sube un archivo .txt o .md para agregarlo como documento"
            )
            
            if uploaded_file is not None:
                try:
                    content = uploaded_file.read().decode('utf-8')
                    if st.button("📥 Importar Archivo"):
                        st.session_state.eval_rag['documents'].append(content)
                        st.session_state.eval_rag['embeddings'] = None
                        st.success(f"Archivo '{uploaded_file.name}' importado exitosamente")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error al leer el archivo: {str(e)}")
            
            st.subheader("🔧 Estado")
            if st.session_state.eval_rag['embeddings'] is not None:
                st.success("✅ Embeddings generados")
            else:
                st.warning("⚠️ Embeddings no generados")
                st.info("Genera embeddings después de modificar documentos (en la pestaña 'Consulta RAG')")

    with tab3:
        st.header("📊 Dashboard de Métricas")
        
        if st.session_state.interaction_logs:
            df = pd.DataFrame([
                {
                    'timestamp': log['timestamp'],
                    'query_length': len(log['query']),
                    'response_length': len(log['response']),
                    **log['metrics']
                }
                for log in st.session_state.interaction_logs
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Response Time")
                fig = px.line(df, x='timestamp', y='total_time', title="Total Time per Query")
                st.plotly_chart(fig, use_container_width=True)
                
                if 'faithfulness' in df.columns:
                    st.subheader("Faithfulness Distribution")
                    fig = px.histogram(df, x='faithfulness', title="Faithfulness Scores Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Retrieval Metrics")
                fig = px.scatter(df, x='retrieval_time', y='generation_time', 
                               size='docs_retrieved', title="Retrieval vs Generation Time")
                st.plotly_chart(fig, use_container_width=True)
                
                if 'relevance' in df.columns and 'context_precision' in df.columns:
                    st.subheader("Relevance vs Context Precision")
                    fig = px.scatter(df, x='relevance', y='context_precision', 
                                   title="Relevance vs Context Precision")
                    st.plotly_chart(fig, use_container_width=True)