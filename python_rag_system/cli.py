#!/usr/bin/env python3

import click
import os
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich.prompt import Prompt, Confirm

from .core.rag_engine import PythonRAGEngine
from .core.llm_integration import LLMCodeAnalyzer
from .core.unified_llm_analyzer import UnifiedLLMCodeAnalyzer
from python_rag_system.core import lang

console = Console()

def create_rag_engine(collection_name: str, persist_dir: str, 
                     embedding_provider: str = "sentence_transformers",
                     embedding_model: str = "all-MiniLM-L6-v2",
                     ollama_url: str = "http://localhost:11434",
                     language: str = "python") -> PythonRAGEngine:
    """Создает RAG движок с указанными параметрами"""
    return PythonRAGEngine(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        ollama_base_url=ollama_url,
        language=language
    )

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🐍 Python RAG System - Система анализа кода с ChromaDB и LLM"""
    pass

@cli.command()
@click.argument('project_path', type=click.Path(exists=True))
@click.option('--collection-name', '-c', default='python_code_rag', help='Имя коллекции ChromaDB')
@click.option('--persist-dir', '-p', default='.chroma_rag_store', help='Директория для хранения ChromaDB')
@click.option('--embedding-provider', type=click.Choice(['sentence_transformers', 'ollama']), 
              default='sentence_transformers', help='Провайдер эмбеддингов')
@click.option('--embedding-model', default='all-MiniLM-L6-v2', help='Модель эмбеддингов')
@click.option('--ollama-url', default='http://localhost:11434', help='URL Ollama сервера')
@click.option('--exclude', '-e', multiple=True, help='Паттерны для исключения файлов')
@click.option('--clear', is_flag=True, help='Очистить существующую коллекцию')
@click.option('--language', default="python", help="Язык проекта: python или javascript")
def index(project_path, collection_name, persist_dir, embedding_provider, embedding_model, ollama_url, exclude, clear, language):
    """Индексирует Python проект в ChromaDB"""
    
    console.print(lang.tr('indexing_project', project_path=project_path))
    
    # Инициализируем RAG движок
    rag_engine = create_rag_engine(
        collection_name=collection_name,
        persist_dir=persist_dir,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        ollama_url=ollama_url,
        language=language
    )
    
    # Очищаем коллекцию если нужно
    if clear:
        if Confirm.ask(lang.tr('confirm_clear_collection')):
            rag_engine.clear_collection()
    
    # Индексируем проект
    exclude_patterns = list(exclude) if exclude else None
    summary = rag_engine.index_project(project_path, exclude_patterns)
    
    if summary:
        # Отображаем статистику
        table = Table(title=lang.tr('indexing_stats_title'))
        table.add_column(lang.tr('metric'), style="cyan")
        table.add_column(lang.tr('value'), style="green")
        
        table.add_row(lang.tr('total_elements'), str(summary['total_elements']))
        table.add_row(lang.tr('functions'), str(summary['functions']))
        table.add_row(lang.tr('classes'), str(summary['classes']))
        table.add_row(lang.tr('imports'), str(summary['imports']))
        table.add_row(lang.tr('files_analyzed'), str(summary['files_analyzed']))
        table.add_row(lang.tr('call_graph_edges'), str(summary['call_graph_edges']))
        
        console.print(table)
        
        # Генерируем документацию
        docs = rag_engine.generate_project_documentation()
        with open("project_documentation.md", "w", encoding="utf-8") as f:
            f.write(docs)
        console.print(lang.tr('indexing_complete', file='project_documentation.md'))
    else:
        console.print(lang.tr('indexing_error'))

@cli.command()
@click.argument('query')
@click.option('--collection-name', '-c', default='python_code_rag', help='Имя коллекции ChromaDB')
@click.option('--persist-dir', '-p', default='.chroma_rag_store', help='Директория ChromaDB')
@click.option('--embedding-provider', type=click.Choice(['sentence_transformers', 'ollama']), 
              default='sentence_transformers', help='Провайдер эмбеддингов')
@click.option('--embedding-model', default='all-MiniLM-L6-v2', help='Модель эмбеддингов')
@click.option('--ollama-url', default='http://localhost:11434', help='URL Ollama сервера')
@click.option('--limit', '-l', default=5, help='Количество результатов')
@click.option('--type-filter', '-t', help='Фильтр по типу (function, class, import)')
@click.option('--file-filter', '-f', help='Фильтр по файлу')
def search(query, collection_name, persist_dir, embedding_provider, embedding_model, ollama_url, limit, type_filter, file_filter):
    """Поиск по коду"""
    
    console.print(lang.tr('search_query', query=query))
    
    rag_engine = create_rag_engine(
        collection_name=collection_name,
        persist_dir=persist_dir,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        ollama_url=ollama_url
    )
    
    results = rag_engine.search(
        query=query,
        n_results=limit,
        filter_type=type_filter,
        filter_file=file_filter
    )
    
    if not results:
        console.print(lang.tr('no_results_found'))
        return
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        distance = result.get('distance', 0)
        
        panel_title = lang.tr('search_result_title', index=i, name=metadata['name'], type=metadata['type'])
        panel_content = lang.tr('search_result_content', file_path=metadata['file_path'], 
                                line_start=metadata['line_start'], line_end=metadata['line_end'], 
                                relevance=1-distance, complexity=metadata.get('complexity', 'N/A'), 
                                code=result['document'][:300])
        
        console.print(Panel(panel_content, title=panel_title, border_style="blue"))

@cli.command()
@click.argument('function_name')
@click.option('--collection-name', '-c', default='python_code_rag', help='Имя коллекции ChromaDB')
@click.option('--persist-dir', '-p', default='.chroma_rag_store', help='Директория ChromaDB')
@click.option('--llm-provider', type=click.Choice(['openai', 'ollama']), default='ollama', help='Провайдер LLM')
@click.option('--model', '-m', help='Модель LLM (по умолчанию: gemma2:27b для Ollama, gpt-4 для OpenAI)')
@click.option('--ollama-url', default='http://localhost:11434', help='URL Ollama сервера')
def analyze(function_name, collection_name, persist_dir, llm_provider, model, ollama_url):
    """Анализирует функцию с помощью LLM"""
    
    console.print(lang.tr('analyze_function', function_name=function_name))
    
    rag_engine = PythonRAGEngine(
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    analyzer = UnifiedLLMCodeAnalyzer(
        rag_engine, 
        llm_provider=llm_provider,
        model=model,
        ollama_base_url=ollama_url
    )
    result = analyzer.analyze_function(function_name)
    
    if 'error' in result:
        console.print(lang.tr('analysis_error', error=result['error']))
    else:
        analyzer.display_analysis_result(result, "function")

@cli.command()
@click.argument('function_name')
@click.option('--collection-name', '-c', default='python_code_rag', help='Имя коллекции ChromaDB')
@click.option('--persist-dir', '-p', default='.chroma_rag_store', help='Директория ChromaDB')
@click.option('--llm-provider', type=click.Choice(['openai', 'ollama']), default='ollama', help='Провайдер LLM')
@click.option('--model', '-m', help='Модель LLM')
@click.option('--ollama-url', default='http://localhost:11434', help='URL Ollama сервера')
def flow(function_name, collection_name, persist_dir, llm_provider, model, ollama_url):
    """Анализирует flow выполнения кода"""
    
    console.print(lang.tr('analyze_flow', function_name=function_name))
    
    rag_engine = PythonRAGEngine(
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    analyzer = UnifiedLLMCodeAnalyzer(
        rag_engine, 
        llm_provider=llm_provider,
        model=model,
        ollama_base_url=ollama_url
    )
    result = analyzer.explain_code_flow(function_name)
    
    analyzer.display_analysis_result(result, "flow")

@cli.command()
@click.argument('question')
@click.option('--collection-name', '-c', default='python_code_rag', help='Имя коллекции ChromaDB')
@click.option('--persist-dir', '-p', default='.chroma_rag_store', help='Директория ChromaDB')
@click.option('--llm-provider', type=click.Choice(['openai', 'ollama']), default='ollama', help='Провайдер LLM')
@click.option('--model', '-m', help='Модель LLM')
@click.option('--ollama-url', default='http://localhost:11434', help='URL Ollama сервера')
@click.option('--context', help='Дополнительный контекст для поиска')
def ask(question, collection_name, persist_dir, llm_provider, model, ollama_url, context):
    """Задает вопрос о коде"""
    
    console.print(lang.tr('ask_question', question=question))
    
    rag_engine = PythonRAGEngine(
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    analyzer = UnifiedLLMCodeAnalyzer(
        rag_engine, 
        llm_provider=llm_provider,
        model=model,
        ollama_base_url=ollama_url
    )
    result = analyzer.answer_code_question(question, context)
    
    analyzer.display_analysis_result(result, "question")

@cli.command()
@click.argument('function_name')
@click.option('--collection-name', '-c', default='python_code_rag', help='Имя коллекции ChromaDB')
@click.option('--persist-dir', '-p', default='.chroma_rag_store', help='Директория ChromaDB')
@click.option('--llm-provider', type=click.Choice(['openai', 'ollama']), default='ollama', help='Провайдер LLM')
@click.option('--model', '-m', help='Модель LLM')
@click.option('--ollama-url', default='http://localhost:11434', help='URL Ollama сервера')
def improve(function_name, collection_name, persist_dir, llm_provider, model, ollama_url):
    """Предлагает улучшения для функции"""
    
    console.print(lang.tr('suggest_improvements', function_name=function_name))
    
    rag_engine = PythonRAGEngine(
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    analyzer = UnifiedLLMCodeAnalyzer(
        rag_engine, 
        llm_provider=llm_provider,
        model=model,
        ollama_base_url=ollama_url
    )
    result = analyzer.suggest_improvements(function_name)
    
    analyzer.display_analysis_result(result, "improvements")

@cli.command()
@click.argument('function_name')
@click.option('--collection-name', '-c', default='python_code_rag', help='Имя коллекции ChromaDB')
@click.option('--persist-dir', '-p', default='.chroma_rag_store', help='Директория ChromaDB')
@click.option('--llm-provider', type=click.Choice(['openai', 'ollama']), default='ollama', help='Провайдер LLM')
@click.option('--model', '-m', help='Модель LLM')
@click.option('--ollama-url', default='http://localhost:11434', help='URL Ollama сервера')
def similar(function_name, collection_name, persist_dir, llm_provider, model, ollama_url):
    """Находит похожие функции"""
    
    console.print(lang.tr('find_similar', function_name=function_name))
    
    rag_engine = PythonRAGEngine(
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    analyzer = UnifiedLLMCodeAnalyzer(
        rag_engine, 
        llm_provider=llm_provider,
        model=model,
        ollama_base_url=ollama_url
    )
    result = analyzer.find_similar_functions(function_name)
    
    analyzer.display_analysis_result(result, "similar")

@cli.command()
@click.option('--collection-name', '-c', default='python_code_rag', help='Имя коллекции ChromaDB')
@click.option('--persist-dir', '-p', default='.chroma_rag_store', help='Директория ChromaDB')
def stats(collection_name, persist_dir):
    """Показывает статистику коллекции"""
    
    console.print(lang.tr('collection_stats'))
    
    rag_engine = PythonRAGEngine(
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    stats = rag_engine.get_collection_stats()
    
    table = Table(title=lang.tr('chroma_stats_title'))
    table.add_column(lang.tr('parameter'), style="cyan")
    table.add_column(lang.tr('value'), style="green")
    
    table.add_row(lang.tr('collection_name'), stats['collection_name'])
    table.add_row(lang.tr('total_documents'), str(stats['total_documents']))
    table.add_row(lang.tr('persist_directory'), stats['persist_directory'])
    table.add_row(lang.tr('embedding_model'), stats['embedding_model'])
    
    console.print(table)

@cli.command()
@click.argument('file_path')
@click.option('--collection-name', '-c', default='python_code_rag', help='Имя коллекции ChromaDB')
@click.option('--persist-dir', '-p', default='.chroma_rag_store', help='Директория ChromaDB')
def file_summary(file_path, collection_name, persist_dir):
    """Показывает сводку по файлу"""
    
    console.print(lang.tr('file_summary', file_path=file_path))
    
    rag_engine = PythonRAGEngine(
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    summary = rag_engine.get_file_summary(file_path)
    
    if summary['total_elements'] == 0:
        console.print(lang.tr('file_not_found'))
        return
    
    # Общая статистика
    table = Table(title=lang.tr('file_stats_title', file_path=file_path))
    table.add_column(lang.tr('type'), style="cyan")
    table.add_column(lang.tr('count'), style="green")
    
    table.add_row(lang.tr('total_elements'), str(summary['total_elements']))
    table.add_row(lang.tr('functions'), str(len(summary['functions'])))
    table.add_row(lang.tr('classes'), str(len(summary['classes'])))
    table.add_row(lang.tr('imports'), str(len(summary['imports'])))
    
    console.print(table)
    
    # Детали функций
    if summary['functions']:
        console.print(lang.tr('functions_details'))
        for func in summary['functions'][:10]:  # Показываем первые 10
            metadata = func['metadata']
            console.print(lang.tr('function_detail', name=metadata['name'], 
                                  line_start=metadata['line_start'], line_end=metadata['line_end'], 
                                  complexity=metadata['complexity']))
    
    # Детали классов
    if summary['classes']:
        console.print(lang.tr('classes_details'))
        for cls in summary['classes']:
            metadata = cls['metadata']
            console.print(lang.tr('class_detail', name=metadata['name'], 
                                  line_start=metadata['line_start'], line_end=metadata['line_end']))

@cli.command()
@click.option('--collection-name', '-c', default='python_code_rag', help='Имя коллекции ChromaDB')
@click.option('--persist-dir', '-p', default='.chroma_rag_store', help='Директория ChromaDB')
@click.option('--embedding-provider', type=click.Choice(['sentence_transformers', 'ollama']), 
              default='sentence_transformers', help='Провайдер эмбеддингов')
@click.option('--embedding-model', default='all-MiniLM-L6-v2', help='Модель эмбеддингов')
@click.option('--llm-provider', type=click.Choice(['openai', 'ollama']), default='ollama', help='Провайдер LLM')
@click.option('--llm-model', help='Модель LLM')
@click.option('--ollama-url', default='http://localhost:11434', help='URL Ollama сервера')
def interactive(collection_name, persist_dir, embedding_provider, embedding_model, llm_provider, llm_model, ollama_url):
    """Интерактивный режим работы с RAG системой"""
    
    console.print(lang.tr('interactive_mode'))
    console.print(lang.tr('available_commands'))
    
    rag_engine = create_rag_engine(
        collection_name=collection_name,
        persist_dir=persist_dir,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        ollama_url=ollama_url
    )
    
    analyzer = UnifiedLLMCodeAnalyzer(
        rag_engine, 
        llm_provider=llm_provider,
        model=llm_model,
        ollama_base_url=ollama_url
    )
    
    while True:
        try:
            command = Prompt.ask(lang.tr('command_prompt')).strip().lower()
            
            if command == 'quit' or command == 'exit':
                console.print(lang.tr('goodbye'))
                break
            
            elif command == 'search':
                query = Prompt.ask(lang.tr('search_query_prompt'))
                results = rag_engine.search(query, n_results=3)
                
                for i, result in enumerate(results, 1):
                    metadata = result['metadata']
                    console.print(lang.tr('search_result', index=i, name=metadata['name'], 
                                          type=metadata['type'], file_path=metadata['file_path'], 
                                          code=result['document'][:200]))
            
            elif command == 'analyze':
                function_name = Prompt.ask(lang.tr('analyze_function_prompt'))
                result = analyzer.analyze_function(function_name)
                analyzer.display_analysis_result(result, "function")
            
            elif command == 'flow':
                function_name = Prompt.ask(lang.tr('analyze_flow_prompt'))
                result = analyzer.explain_code_flow(function_name)
                analyzer.display_analysis_result(result, "flow")
            
            elif command == 'ask':
                question = Prompt.ask(lang.tr('ask_question_prompt'))
                result = analyzer.answer_code_question(question)
                analyzer.display_analysis_result(result, "question")
            
            elif command == 'improve':
                function_name = Prompt.ask(lang.tr('improve_function_prompt'))
                result = analyzer.suggest_improvements(function_name)
                analyzer.display_analysis_result(result, "improvements")
            
            elif command == 'similar':
                function_name = Prompt.ask(lang.tr('similar_function_prompt'))
                result = analyzer.find_similar_functions(function_name)
                analyzer.display_analysis_result(result, "similar")
            
            elif command == 'stats':
                stats = rag_engine.get_collection_stats()
                console.print(lang.tr('collection_stats_summary', total_documents=stats['total_documents']))
            
            else:
                console.print(lang.tr('unknown_command'))
        
        except KeyboardInterrupt:
            console.print(lang.tr('interrupted'))
            break
        except Exception as e:
            console.print(lang.tr('error', error=str(e)))

if __name__ == '__main__':
    cli()