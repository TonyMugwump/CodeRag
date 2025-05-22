import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os

class ChromaRAG:
    """
    Класс для хранения и поиска семантических и flow RAG-структур в ChromaDB.

    - code_blocks: коллекция для функций, классов, модулей (AST-блоки).
    - flows: коллекция для flows, сценариев, entry points, data entities.

    Метаданные и связи хранятся в виде полей в коллекциях.
    """
    def __init__(self, persist_dir=".chroma_rag_store"):
        """
        Инициализация клиента ChromaDB и коллекций.
        :param persist_dir: Путь к директории для хранения данных ChromaDB.
        """
        self.client = chromadb.PersistentClient(path=persist_dir)
        # Коллекция для кода (функции, классы, модули)
        self.code_collection = self.client.get_or_create_collection("code_blocks")
        # Коллекция для flows/scenarios
        self.flow_collection = self.client.get_or_create_collection("flows")

    def add_code_block(self, block_id, embedding, metadata):
        """
        Добавить один блок кода (функция, класс, модуль) в коллекцию code_blocks.
        :param block_id: str, уникальный идентификатор блока
        :param embedding: list[float], эмбеддинг блока
        :param metadata: dict, метаданные (имя, файл, docstring, связи и т.д.)
        """
        self.code_collection.add(
            ids=[block_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[metadata.get("docstring", "")]
        )

    def add_flow(self, flow_id, embedding, metadata):
        """
        Добавить один flow/scenario/entry point в коллекцию flows.
        :param flow_id: str, уникальный идентификатор flow
        :param embedding: list[float], эмбеддинг flow
        :param metadata: dict, метаданные (описание, связи и т.д.)
        """
        self.flow_collection.add(
            ids=[flow_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[metadata.get("description", "")]
        )

    def query_code(self, query_embedding, n_results=5, where=None):
        """
        Семантический поиск по коллекции code_blocks.
        :param query_embedding: list[float], эмбеддинг запроса
        :param n_results: int, количество результатов
        :param where: dict, фильтр по метаданным
        :return: dict с результатами поиска
        """
        return self.code_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

    def query_flows(self, query_embedding, n_results=5, where=None):
        """
        Семантический поиск по коллекции flows.
        :param query_embedding: list[float], эмбеддинг запроса
        :param n_results: int, количество результатов
        :param where: dict, фильтр по метаданным
        :return: dict с результатами поиска
        """
        return self.flow_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

    def get_code_block(self, block_id):
        """
        Получить блок кода по id.
        :param block_id: str, идентификатор блока
        :return: dict с данными блока
        """
        return self.code_collection.get(ids=[block_id])

    def get_flow(self, flow_id):
        """
        Получить flow/scenario по id.
        :param flow_id: str, идентификатор flow
        :return: dict с данными flow
        """
        return self.flow_collection.get(ids=[flow_id])

    def add_code_blocks_bulk(self, ids, embeddings, metadatas, documents=None):
        """
        Добавить несколько блоков кода в коллекцию code_blocks.
        :param ids: list[str], идентификаторы
        :param embeddings: list[list[float]], эмбеддинги
        :param metadatas: list[dict], метаданные
        :param documents: list[str], docstring (опционально)
        """
        self.code_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents if documents else [m.get("docstring", "") for m in metadatas]
        )
    def add_flows_bulk(self, ids, embeddings, metadatas, documents=None):
        """
        Добавить несколько flows/scenarios в коллекцию flows.
        :param ids: list[str], идентификаторы
        :param embeddings: list[list[float]], эмбеддинги
        :param metadatas: list[dict], метаданные
        :param documents: list[str], описания (опционально)
        """
        self.flow_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents if documents else [m.get("description", "") for m in metadatas]
        )
