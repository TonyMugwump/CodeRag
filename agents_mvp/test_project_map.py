from project_map import collect_functions_and_calls, generate_mermaid_call_graph
from rag_chroma import ChromaRAG
import numpy as np
import shutil
import os

if __name__ == "__main__":
    project_path = "/home/quit/Documents/CodeRag"  # или другой путь к проекту
    function_map, call_graph = collect_functions_and_calls(project_path)
    mermaid_code = generate_mermaid_call_graph(function_map, call_graph)
    with open("call_graph.mmd", "w") as f:
        f.write(mermaid_code)
    print("Mermaid call graph saved to call_graph.mmd")
    print(f"Functions found: {len(function_map)}")
    print(f"Call graph edges: {sum(len(v) for v in call_graph.values())}")

    # --- Тест ChromaRAG ---
    print("\n--- ChromaRAG TEST ---")
    persist_dir = ".chroma_rag_test_store/chroma"
    # Удалим старую тестовую директорию или файл, если есть
    if os.path.exists(persist_dir):
        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir)
        else:
            os.remove(persist_dir)
    rag = ChromaRAG(persist_dir=persist_dir)

    # Добавим тестовые code_blocks
    ids = ["func1", "func2", "func3"]
    embeddings = [np.random.rand(384).tolist() for _ in ids]  # 384 - размерность для miniLM
    metadatas = [
        {"name": "func1", "file": "a.py", "docstring": "Test function 1"},
        {"name": "func2", "file": "b.py", "docstring": "Test function 2"},
        {"name": "func3", "file": "c.py", "docstring": "Test function 3"},
    ]
    rag.add_code_blocks_bulk(ids, embeddings, metadatas)
    print("Added code_blocks:", ids)

    # Добавим тестовые flows
    flow_ids = ["flow1", "flow2"]
    flow_embeddings = [np.random.rand(384).tolist() for _ in flow_ids]
    flow_metadatas = [
        {"description": "Main flow", "entry": "main.py"},
        {"description": "Aux flow", "entry": "aux.py"},
    ]
    rag.add_flows_bulk(flow_ids, flow_embeddings, flow_metadatas)
    print("Added flows:", flow_ids)

    # Семантический поиск по code_blocks (по случайному эмбеддингу)
    query_emb = np.random.rand(384).tolist()
    code_results = rag.query_code(query_emb, n_results=2)
    print("\nCode search results:")
    for i, doc in enumerate(code_results['documents']):
        print(f"Result {i+1}: {doc}")

    # Семантический поиск по flows
    flow_results = rag.query_flows(query_emb, n_results=2)
    print("\nFlow search results:")
    for i, doc in enumerate(flow_results['documents']):
        print(f"Result {i+1}: {doc}")

    # Очистка тестовой директории или файла
    if os.path.exists(persist_dir):
        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir)
        else:
            os.remove(persist_dir)
    print("ChromaRAG test completed and cleaned up.") 