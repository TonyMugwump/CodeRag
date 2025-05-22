import chromadb
from chromadb.config import Settings
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import os
from pathlib import Path
import tiktoken
from rich.console import Console
from rich.progress import Progress, TaskID

from .code_parser import PythonCodeParser, CodeElement
from .embeddings import EmbeddingFactory, EmbeddingProvider

console = Console()

class PythonRAGEngine:
    """RAG движок для Python проектов с ChromaDB"""
    
    def __init__(self, 
                 collection_name: str = "python_code_rag",
                 persist_directory: str = ".chroma_rag_store",
                 embedding_provider: str = "sentence_transformers",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 ollama_base_url: str = "http://localhost:11434"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_provider_type = embedding_provider
        self.embedding_model_name = embedding_model
        self.ollama_base_url = ollama_base_url
        
        # Инициализируем ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Инициализируем провайдер эмбеддингов
        self.embedding_provider = EmbeddingFactory.create_provider(
            embedding_provider,
            model_name=embedding_model,
            base_url=ollama_base_url
        )
        
        # Создаем кастомную функцию эмбеддингов для ChromaDB
        class CustomEmbeddingFunction:
            def __init__(self, provider):
                self.provider = provider
            
            def __call__(self, input):
                # ChromaDB ожидает именно параметр 'input'
                embeddings = self.provider.encode(input)
                return embeddings.tolist()
        
        self.embedding_function = CustomEmbeddingFunction(self.embedding_provider)
        
        # Создаем или получаем коллекцию
        try:
            # Сначала пытаемся получить существующую коллекцию
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if collection_name in existing_collections:
                # Коллекция существует, получаем её
                try:
                    self.collection = self.client.get_collection(collection_name)
                    console.print(f"[green]Загружена существующая коллекция: {collection_name}[/green]")
                    
                    # Проверяем совместимость размерности эмбеддингов
                    expected_dim = self.embedding_provider.get_dimension()
                    
                    # Пытаемся сделать тестовый поиск для проверки совместимости
                    try:
                        self.collection.query(query_texts=["test"], n_results=1)
                    except Exception as dim_error:
                        if "dimension" in str(dim_error).lower():
                            console.print(f"[yellow]Несовместимость размерности эмбеддингов. Пересоздаю коллекцию...[/yellow]")
                            self.client.delete_collection(collection_name)
                            raise Exception("Dimension mismatch")
                        else:
                            raise dim_error
                            
                except Exception:
                    # Если не удалось загрузить или есть проблемы с размерностью, создаем новую
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        metadata={
                            "description": "Python code RAG collection",
                            "embedding_provider": embedding_provider,
                            "embedding_model": embedding_model
                        },
                        embedding_function=self.embedding_function
                    )
                    console.print(f"[blue]Пересоздана коллекция: {collection_name}[/blue]")
            else:
                # Коллекции нет, создаем новую
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "description": "Python code RAG collection",
                        "embedding_provider": embedding_provider,
                        "embedding_model": embedding_model
                    },
                    embedding_function=self.embedding_function
                )
                console.print(f"[blue]Создана новая коллекция: {collection_name}[/blue]")
                
        except Exception as e:
            console.print(f"[red]Ошибка при работе с коллекцией: {e}[/red]")
            # Если что-то пошло не так, пытаемся создать коллекцию с уникальным именем
            import time
            unique_name = f"{collection_name}_{int(time.time())}"
            self.collection = self.client.create_collection(
                name=unique_name,
                metadata={
                    "description": "Python code RAG collection",
                    "embedding_provider": embedding_provider,
                    "embedding_model": embedding_model
                },
                embedding_function=self.embedding_function
            )
            console.print(f"[yellow]Создана коллекция с уникальным именем: {unique_name}[/yellow]")
        
        # Парсер кода
        self.parser = PythonCodeParser()
        
        # Токенизатор для подсчета токенов
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def index_project(self, project_path: str, exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """Индексирует весь Python проект"""
        console.print(f"[yellow]Начинаю индексацию проекта: {project_path}[/yellow]")
        
        # Парсим проект
        project_elements = self.parser.parse_project(project_path, exclude_patterns)
        
        if not project_elements:
            console.print("[red]Не найдено Python файлов для индексации[/red]")
            return {}
        
        total_elements = sum(len(elements) for elements in project_elements.values())
        
        with Progress() as progress:
            task = progress.add_task("[green]Индексация элементов...", total=total_elements)
            
            documents = []
            metadatas = []
            ids = []
            
            for file_path, elements in project_elements.items():
                for element in elements:
                    # Создаем документ для индексации
                    doc_text = self._create_document_text(element)
                    metadata = self._create_metadata(element, file_path)
                    doc_id = str(uuid.uuid4())
                    
                    documents.append(doc_text)
                    metadatas.append(metadata)
                    ids.append(doc_id)
                    
                    progress.advance(task)
            
            # Добавляем в ChromaDB
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
        
        # Сохраняем граф вызовов
        self._save_call_graph()
        
        summary = self.parser.get_project_summary()
        console.print(f"[green]Индексация завершена! Проиндексировано {total_elements} элементов[/green]")
        
        return summary
    
    def _create_document_text(self, element: CodeElement) -> str:
        """Создает текст документа для индексации"""
        parts = []
        
        # Основная информация
        parts.append(f"Type: {element.type}")
        parts.append(f"Name: {element.name}")
        parts.append(f"File: {element.file_path}")
        
        # Docstring если есть
        if element.docstring:
            parts.append(f"Documentation: {element.docstring}")
        
        # Исходный код
        parts.append(f"Source code:\n{element.source_code}")
        
        # Зависимости и вызовы
        if element.calls:
            parts.append(f"Calls: {', '.join(element.calls)}")
        
        if element.dependencies:
            parts.append(f"Dependencies: {', '.join(element.dependencies)}")
        
        # Дополнительная информация для функций
        if element.type in ['function', 'async_function']:
            parts.append(f"Complexity: {element.complexity}")
            
            # Получаем flow информацию только для функций
            try:
                flow_info = self.parser.get_function_flow(element.name)
                if flow_info:
                    if flow_info.get('calls'):
                        parts.append(f"Function calls: {', '.join([c.split('::')[-1] for c in flow_info['calls']])}")
                    if flow_info.get('called_by'):
                        parts.append(f"Called by: {', '.join([c.split('::')[-1] for c in flow_info['called_by']])}")
            except Exception as e:
                # Если возникла ошибка при получении flow информации, просто пропускаем
                console.print(f"[yellow]Warning: Could not get flow info for {element.name}: {e}[/yellow]")
                pass
        
        # Информация о классах
        if element.type == 'class' and element.children:
            parts.append(f"Methods: {', '.join(element.children)}")
        
        return "\n".join(parts)
    
    def _create_metadata(self, element: CodeElement, file_path: str) -> Dict[str, Any]:
        """Создает метаданные для элемента"""
        metadata = {
            'name': element.name,
            'type': element.type,
            'file_path': file_path,
            'line_start': element.line_start,
            'line_end': element.line_end,
            'complexity': element.complexity,
            'has_docstring': bool(element.docstring),
            'calls_count': len(element.calls),
            'dependencies_count': len(element.dependencies)
        }
        
        # Добавляем информацию о родителе для методов
        if element.parent:
            metadata['parent'] = element.parent
        
        # Добавляем количество детей для классов
        if element.children:
            metadata['children_count'] = len(element.children)
        
        return metadata
    
    def search(self, 
               query: str, 
               n_results: int = 10,
               filter_type: Optional[str] = None,
               filter_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Поиск по коду"""
        
        # Создаем фильтр
        where_filter = {}
        if filter_type:
            where_filter['type'] = filter_type
        if filter_file:
            where_filter['file_path'] = {"$contains": filter_file}
        
        # Выполняем поиск
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        # Форматируем результаты
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                result = {
                    'document': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'id': results['ids'][0][i]
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_function_context(self, function_name: str, include_callers: bool = True, include_callees: bool = True) -> Dict[str, Any]:
        """Получает полный контекст функции включая вызовы"""
        
        # Ищем функцию
        results = self.search(f"name:{function_name}", filter_type="function")
        if not results:
            results = self.search(f"name:{function_name}", filter_type="async_function")
        
        if not results:
            return {"error": f"Функция {function_name} не найдена"}
        
        main_function = results[0]
        context = {
            'main_function': main_function,
            'callers': [],
            'callees': [],
            'related_classes': [],
            'flow_diagram': None
        }
        
        # Получаем flow информацию из парсера
        flow_info = self.parser.get_function_flow(function_name)
        if flow_info:
            context['flow_info'] = flow_info
            
            # Ищем функции, которые вызывает данная функция
            if include_callees and flow_info.get('calls'):
                for called_func in flow_info['calls']:
                    func_name = called_func.split('::')[-1]
                    callees = self.search(f"name:{func_name}")
                    context['callees'].extend(callees[:2])  # Ограничиваем количество
            
            # Ищем функции, которые вызывают данную функцию
            if include_callers and flow_info.get('called_by'):
                for caller_func in flow_info['called_by']:
                    func_name = caller_func.split('::')[-1]
                    callers = self.search(f"name:{func_name}")
                    context['callers'].extend(callers[:2])  # Ограничиваем количество
        
        return context
    
    def get_class_context(self, class_name: str) -> Dict[str, Any]:
        """Получает полный контекст класса"""
        
        # Ищем класс
        results = self.search(f"name:{class_name}", filter_type="class")
        if not results:
            return {"error": f"Класс {class_name} не найден"}
        
        main_class = results[0]
        context = {
            'main_class': main_class,
            'methods': [],
            'related_functions': []
        }
        
        # Ищем методы класса
        file_path = main_class['metadata']['file_path']
        methods = self.search(f"file:{file_path}", filter_type="function")
        
        # Фильтруем методы, которые принадлежат этому классу
        for method in methods:
            if class_name.lower() in method['document'].lower():
                context['methods'].append(method)
        
        return context
    
    def get_file_summary(self, file_path: str) -> Dict[str, Any]:
        """Получает сводку по файлу"""
        
        results = self.search(f"file:{file_path}")
        
        summary = {
            'file_path': file_path,
            'functions': [],
            'classes': [],
            'imports': [],
            'total_elements': len(results)
        }
        
        for result in results:
            element_type = result['metadata']['type']
            if element_type in ['function', 'async_function']:
                summary['functions'].append(result)
            elif element_type == 'class':
                summary['classes'].append(result)
            elif element_type == 'import':
                summary['imports'].append(result)
        
        return summary
    
    def _save_call_graph(self):
        """Сохраняет граф вызовов в Mermaid формате"""
        mermaid_graph = self.parser.export_call_graph_mermaid()
        
        with open("call_graph.mmd", "w", encoding="utf-8") as f:
            f.write(mermaid_graph)
        
        console.print("[blue]Граф вызовов сохранен в call_graph.mmd[/blue]")
    
    def generate_project_documentation(self) -> str:
        """Генерирует документацию по проекту"""
        summary = self.parser.get_project_summary()
        
        doc_parts = []
        doc_parts.append("# Документация проекта\n")
        doc_parts.append(f"**Всего элементов:** {summary['total_elements']}")
        doc_parts.append(f"**Функций:** {summary['functions']}")
        doc_parts.append(f"**Классов:** {summary['classes']}")
        doc_parts.append(f"**Импортов:** {summary['imports']}")
        doc_parts.append(f"**Файлов проанализировано:** {summary['files_analyzed']}")
        doc_parts.append(f"**Связей в графе вызовов:** {summary['call_graph_edges']}")
        doc_parts.append(f"**Зависимостей:** {summary['dependency_graph_edges']}\n")
        
        # Добавляем топ функций по сложности
        complex_functions = []
        for element in self.parser.elements.values():
            if element.type in ['function', 'async_function'] and element.complexity > 0:
                complex_functions.append((element.name, element.complexity, element.file_path))
        
        complex_functions.sort(key=lambda x: x[1], reverse=True)
        
        if complex_functions:
            doc_parts.append("## Самые сложные функции\n")
            for name, complexity, file_path in complex_functions[:10]:
                doc_parts.append(f"- **{name}** (сложность: {complexity}) - `{file_path}`")
        
        return "\n".join(doc_parts)
    
    def clear_collection(self):
        """Очищает коллекцию"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Python code RAG collection"}
        )
        console.print("[yellow]Коллекция очищена[/yellow]")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Получает статистику коллекции"""
        count = self.collection.count()
        
        return {
            'total_documents': count,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
            'embedding_provider': self.embedding_provider_type,
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_provider.get_dimension()
        } 