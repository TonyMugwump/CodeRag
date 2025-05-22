import os
from rag_chroma import ChromaRAG
from project_map import collect_functions_and_calls
# from chromadb.utils import embedding_functions  # если нужен стандартный эмбеддер
from rag_index_faiss import ASTBlockExtractor  # временно, потом вынести

class ProjectContext:
    def __init__(self, code_dir, chroma_rag, embedder):
        self.code_dir = code_dir
        self.chroma_rag = chroma_rag
        self.embedder = embedder  # функция: List[str] -> List[List[float]]
        self.file_mtimes = {}
        self.function_map = {}
        self.call_graph = {}
        self.code_ids = []
        self.code_texts = []
        self.code_metas = []
        self._init_full_index()

    def _init_full_index(self):
        self.file_mtimes = {}
        code_ids, code_texts, code_metas = [], [], []
        for root, _, files in os.walk(self.code_dir):
            for fname in files:
                if fname.endswith('.py'):
                    path = os.path.join(root, fname)
                    self.file_mtimes[path] = os.path.getmtime(path)
                    # ASTBlockExtractor.extract_ast_nodes(path) должен возвращать (block_id, code, meta)
                    for block_id, code, meta in ASTBlockExtractor.extract_ast_nodes(path):
                        code_ids.append(block_id)
                        code_texts.append(code)
                        code_metas.append(meta)
        self.code_ids = code_ids
        self.code_texts = code_texts
        self.code_metas = code_metas
        print(f"[ChromaRAG] Найдено {len(code_ids)} блоков кода для индексации.")
        if code_texts:
            code_embeddings = self.embedder(code_texts)
            self.chroma_rag.add_code_blocks_bulk(code_ids, code_embeddings, code_metas)
            print(f"[ChromaRAG] Добавлено {len(code_ids)} блоков в ChromaDB.")
        else:
            print("[ChromaRAG] Нет блоков для добавления в ChromaDB.")
        self.function_map, self.call_graph = collect_functions_and_calls(self.code_dir)

    def update(self):
        current_files = set()
        changed_files = []
        for root, _, files in os.walk(self.code_dir):
            for fname in files:
                if fname.endswith('.py'):
                    path = os.path.join(root, fname)
                    current_files.add(path)
                    mtime = os.path.getmtime(path)
                    if path not in self.file_mtimes or self.file_mtimes[path] != mtime:
                        changed_files.append(path)
                        self.file_mtimes[path] = mtime
        deleted_files = set(self.file_mtimes) - current_files
        # (Для простоты MVP) пока не реализуем удаление из ChromaDB, только добавление/обновление
        code_ids, code_texts, code_metas = [], [], []
        for path in changed_files:
            from rag_index_faiss import ASTBlockExtractor
            for block_id, code, meta in ASTBlockExtractor.extract_ast_nodes(path):
                code_ids.append(block_id)
                code_texts.append(code)
                code_metas.append(meta)
            print(f"[ChromaRAG] Обновление: найдено {len(code_ids)} новых/изменённых блоков в {path}.")
            if code_texts:
                code_embeddings = self.embedder(code_texts)
                self.chroma_rag.add_code_blocks_bulk(code_ids, code_embeddings, code_metas)
                print(f"[ChromaRAG] Добавлено {len(code_ids)} блоков в ChromaDB из {path}.")
        # Обновляем общий список метаданных
        self.code_ids = code_ids
        self.code_texts = code_texts
        self.code_metas = code_metas
        self.function_map, self.call_graph = collect_functions_and_calls(self.code_dir)

    def query(self, query_text, top_k=5):
        query_emb = self.embedder([query_text])[0]
        results = self.chroma_rag.query_code(query_emb, n_results=top_k)
        out = []
        for i, block_id in enumerate(results['ids'][0]):
            meta = results['metadatas'][0][i]
            out.append({
                'id': block_id,
                'score': results['distances'][0][i] if 'distances' in results else None,
                'meta': meta
            })
        return out 