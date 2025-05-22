import sys
from architect_mikhail import ArchitectMikhail
from code_agent import CodeAgent, ollama_llm, azure_llm
from test_agent import TestAgent
from extract import extract_text_from_file
from project_map import collect_functions_and_calls
from project_context import ProjectContext
from rag_chroma import ChromaRAG
from rag_index_faiss import azure_get_embeddings
import numpy as np


def main():
    print("=== Agents MVP CLI (ChromaDB, Ollama, Project Map, Incremental Refresh) ===")
    code_dir = "/home/quit/Documents/CodeRag"
    persist_dir = "/home/quit/Documents/Agents/.chroma_rag_store/CodeRag"
    chroma_rag = ChromaRAG(persist_dir=persist_dir)
    #ilya = CodeAgent(lambda prompt: ollama_llm(prompt, model="gemma3:27b"))
    ilya = CodeAgent(azure_llm)
    context = ProjectContext(code_dir, chroma_rag, azure_get_embeddings)
    while True:
        user_query = input("Введите запрос (или 'exit' для выхода): ")
        if user_query.strip().lower() == "exit":
            break
        context.update()  # инкрементальное обновление
        mikhail = ArchitectMikhail(context, ilya, context.function_map, context.call_graph)
        results = mikhail.process_request(user_query)
        print("\n--- Preview изменений ---")
        for block_id, diff in results:
            print(f"\n=== {block_id} ===\n{diff}\n")

if __name__ == "__main__":
    main() 