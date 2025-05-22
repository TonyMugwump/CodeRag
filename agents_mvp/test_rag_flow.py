from project_map import collect_functions_and_calls
from rag_chroma import ChromaRAG
import numpy as np
from grisha import GrishaAgent
import os

# Пример класса FlowRAG из документации
class FlowRAG:
    def __init__(self):
        self.nodes = {}  # node_id -> node_info
        self.edges = []  # (from_id, to_id, relation)

    def add_node(self, node_id, node_type, **attrs):
        self.nodes[node_id] = {"type": node_type, **attrs}

    def add_edge(self, from_id, to_id, relation):
        self.edges.append((from_id, to_id, relation))

def extract_flows_from_docstring(docstring):
    flows = []
    if docstring:
        for line in docstring.splitlines():
            if "flow" in line.lower() or "pipeline" in line.lower() or "scenario" in line.lower():
                flows.append(line.strip())
    return flows

def build_project_flow_rag(project_path):
    function_map, call_graph = collect_functions_and_calls(project_path)
    rag = FlowRAG()
    # 1. Добавляем функции и классы как узлы
    for func_id, info in function_map.items():
        rag.add_node(func_id, "Function", doc=info["doc"], file=info["file"])
        # 2. Извлекаем flows из docstring
        flows = extract_flows_from_docstring(info["doc"])
        for flow in flows:
            flow_id = flow.replace(" ", "_")
            rag.add_node(flow_id, "Flow", description=flow)
            rag.add_edge(flow_id, func_id, "implements")
    # 3. Добавляем связи вызовов
    for caller, callees in call_graph.items():
        for callee in callees:
            rag.add_edge(caller, callee, "calls")
    # 4. (Опционально) ищем точки входа
    # ... (например, main.py:main, app.py:run, etc.)
    return rag

def user_confirm():
    resp = input("В коде нет flows/docstring. Запустить Григория для автодокументации? (y/n): ")
    return resp.strip().lower() == 'y'

if __name__ == "__main__":
    project_path = "/home/quit/Documents/CodeRag"  # путь к проекту
    function_map, call_graph = collect_functions_and_calls(project_path)
    # --- ChromaRAG для flows ---
    persist_dir = ".chroma_rag_store/CodeRag_FlowTest"
    rag = ChromaRAG(persist_dir=persist_dir)
    flow_ids, flow_texts, flow_metadatas = [], [], []
    for func_id, info in function_map.items():
        flows = extract_flows_from_docstring(info["doc"])
        for flow in flows:
            flow_id = flow.replace(" ", "_")
            flow_ids.append(flow_id)
            flow_texts.append(flow)
            flow_metadatas.append({"description": flow, "source_func": func_id})
    if flow_texts:
        # dummy-эмбеддинги, замените на azure_get_embeddings при необходимости
        flow_embeddings = [np.random.rand(384).tolist() for _ in flow_texts]
        rag.add_flows_bulk(flow_ids, flow_embeddings, flow_metadatas)
        print(f"[ChromaRAG] Добавлено {len(flow_ids)} flows в ChromaDB.")
        # Семантический поиск по flows
        query_emb = np.random.rand(384).tolist()
        flow_results = rag.query_flows(query_emb, n_results=3)
        print("\nFlow search results:")
        for i, doc in enumerate(flow_results['documents']):
            print(f"Result {i+1}: {doc}")
    else:
        print("[ChromaRAG] Не найдено flows для добавления в ChromaDB.")
        if user_confirm():
            from project_context import ProjectContext
            from rag_index_faiss import azure_get_embeddings
            from rag_chroma import ChromaRAG
            context = ProjectContext(project_path, ChromaRAG(persist_dir=persist_dir), azure_get_embeddings)
            grisha = GrishaAgent(llm_model=os.getenv("AZURE_LLM_MODEL"))
            stats = grisha.generate_docstrings_and_update(context)
            print(f"[Grisha] {stats}")
            # Повторяем попытку построения flows
            function_map, call_graph = collect_functions_and_calls(project_path)
            flow_ids, flow_texts, flow_metadatas = [], [], []
            for func_id, info in function_map.items():
                flows = extract_flows_from_docstring(info["doc"])
                for flow in flows:
                    flow_id = flow.replace(" ", "_")
                    flow_ids.append(flow_id)
                    flow_texts.append(flow)
                    flow_metadatas.append({"description": flow, "source_func": func_id})
            if flow_texts:
                flow_embeddings = [np.random.rand(384).tolist() for _ in flow_texts]
                rag.add_flows_bulk(flow_ids, flow_embeddings, flow_metadatas)
                print(f"[ChromaRAG] После Гриши добавлено {len(flow_ids)} flows в ChromaDB.")
            else:
                print("[ChromaRAG] Даже после автодокументации не найдено flows.")
    # Выводим все flows из ChromaDB
    all_flows = rag.flow_collection.get()
    print("Flows in ChromaDB:")
    for i, flow in enumerate(all_flows['ids']):
        print(f"  {flow}: {all_flows['documents'][i]}")