import ast
import os
from typing import List, Tuple, Dict
import numpy as np
import requests
import faiss
from dotenv import load_dotenv


load_dotenv()


# --- Azure Embedding Function ---
def azure_get_embeddings(texts: List[str]) -> np.ndarray:
    endpoint = os.getenv("AZURE_ENDPOINT")
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
    api_version = os.getenv("AZURE_API_VERSION")
    api_key = os.getenv("AZURE_API_KEY")

    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    data = {
        "input": texts
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    # Azure возвращает список объектов с ключом 'embedding'
    embeddings = [item["embedding"] for item in result["data"]]
    return np.array(embeddings, dtype=np.float32)

class ASTBlockExtractor:
    @staticmethod
    def extract_ast_nodes(filepath: str) -> List[Tuple[str, str, Dict]]:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        blocks = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
                start_line = node.lineno - 1
                end_line = getattr(node, 'end_lineno', start_line + 1)
                lines = source.splitlines()[start_line:end_line]
                block_code = "\n".join(lines)
                metadata = {
                    "type": type(node).__name__,
                    "name": name,
                    "filepath": filepath,
                    "start_line": start_line + 1,
                    "end_line": end_line
                }
                blocks.append((f"{filepath}:{name}", block_code, metadata))
        return blocks

class RAGIndexFAISS:
    def __init__(self, code_directory: str):
        self.code_directory = code_directory
        self.ids, self.texts, self.metas = self.collect_code_blocks(code_directory)
        self.embeddings = azure_get_embeddings(self.texts)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def collect_code_blocks(self, directory: str) -> Tuple[List[str], List[str], List[Dict]]:
        ids, texts, metadatas = [], [], [];
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    for block_id, code, meta in ASTBlockExtractor.extract_ast_nodes(full_path):
                        ids.append(block_id)
                        texts.append(code)
                        metadatas.append(meta)
        return ids, texts, metadatas

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        query_emb = azure_get_embeddings([query_text])
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for idx, i in enumerate(indices[0]):
            result = {
                "id": self.ids[i],
                "score": float(distances[0][idx]),
                "meta": self.metas[i]
            }
            results.append(result)
        return results 