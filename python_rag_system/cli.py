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

console = Console()

def create_rag_engine(collection_name: str, persist_dir: str, 
                     embedding_provider: str = "sentence_transformers",
                     embedding_model: str = "all-MiniLM-L6-v2",
                     ollama_url: str = "http://localhost:11434") -> PythonRAGEngine:
    """Создает RAG движок с указанными параметрами"""
    return PythonRAGEngine(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        ollama_base_url=ollama_url
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
def index(project_path, collection_name, persist_dir, embedding_provider, embedding_model, ollama_url, exclude, clear):
    """Индексирует Python проект в ChromaDB"""
    
    console.print(f"[bold blue]🚀 Индексация проекта: {project_path}[/bold blue]")
    
    # Инициализируем RAG движок
    rag_engine = create_rag_engine(
        collection_name=collection_name,
        persist_dir=persist_dir,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        ollama_url=ollama_url
    )
    
    # Очищаем коллекцию если нужно
    if clear:
        if Confirm.ask("Вы уверены, что хотите очистить существующую коллекцию?"):
            rag_engine.clear_collection()
    
    # Индексируем проект
    exclude_patterns = list(exclude) if exclude else None
    summary = rag_engine.index_project(project_path, exclude_patterns)
    
    if summary:
        # Отображаем статистику
        table = Table(title="Статистика индексации")
        table.add_column("Метрика", style="cyan")
        table.add_column("Значение", style="green")
        
        table.add_row("Всего элементов", str(summary['total_elements']))
        table.add_row("Функций", str(summary['functions']))
        table.add_row("Классов", str(summary['classes']))
        table.add_row("Импортов", str(summary['imports']))
        table.add_row("Файлов", str(summary['files_analyzed']))
        table.add_row("Связей в графе", str(summary['call_graph_edges']))
        
        console.print(table)
        
        # Генерируем документацию
        docs = rag_engine.generate_project_documentation()
        with open("project_documentation.md", "w", encoding="utf-8") as f:
            f.write(docs)
        
        console.print("[green]✅ Индексация завершена! Документация сохранена в project_documentation.md[/green]")
    else:
        console.print("[red]❌ Ошибка индексации[/red]")

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
    
    console.print(f"[bold blue]🔍 Поиск: {query}[/bold blue]")
    
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
        console.print("[yellow]Результаты не найдены[/yellow]")
        return
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        distance = result.get('distance', 0)
        
        panel_title = f"Результат {i}: {metadata['name']} ({metadata['type']})"
        panel_content = f"""
[bold]Файл:[/bold] {metadata['file_path']}
[bold]Строки:[/bold] {metadata['line_start']}-{metadata['line_end']}
[bold]Релевантность:[/bold] {1-distance:.3f}
[bold]Сложность:[/bold] {metadata.get('complexity', 'N/A')}

[bold]Код:[/bold]
{result['document'][:300]}...
        """
        
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
    
    console.print(f"[bold blue]🧠 Анализ функции: {function_name}[/bold blue]")
    
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
    
    console.print(f"[bold blue]🌊 Анализ flow: {function_name}[/bold blue]")
    
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
    
    console.print(f"[bold blue]❓ Вопрос: {question}[/bold blue]")
    
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
    
    console.print(f"[bold blue]⚡ Предложения по улучшению: {function_name}[/bold blue]")
    
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
    
    console.print(f"[bold blue]🔗 Поиск похожих функций: {function_name}[/bold blue]")
    
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
    
    console.print("[bold blue]📊 Статистика коллекции[/bold blue]")
    
    rag_engine = PythonRAGEngine(
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    stats = rag_engine.get_collection_stats()
    
    table = Table(title="Статистика ChromaDB")
    table.add_column("Параметр", style="cyan")
    table.add_column("Значение", style="green")
    
    table.add_row("Коллекция", stats['collection_name'])
    table.add_row("Документов", str(stats['total_documents']))
    table.add_row("Директория", stats['persist_directory'])
    table.add_row("Модель эмбеддингов", stats['embedding_model'])
    
    console.print(table)

@cli.command()
@click.argument('file_path')
@click.option('--collection-name', '-c', default='python_code_rag', help='Имя коллекции ChromaDB')
@click.option('--persist-dir', '-p', default='.chroma_rag_store', help='Директория ChromaDB')
def file_summary(file_path, collection_name, persist_dir):
    """Показывает сводку по файлу"""
    
    console.print(f"[bold blue]📄 Сводка по файлу: {file_path}[/bold blue]")
    
    rag_engine = PythonRAGEngine(
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    summary = rag_engine.get_file_summary(file_path)
    
    if summary['total_elements'] == 0:
        console.print("[yellow]Файл не найден в индексе[/yellow]")
        return
    
    # Общая статистика
    table = Table(title=f"Статистика файла: {file_path}")
    table.add_column("Тип", style="cyan")
    table.add_column("Количество", style="green")
    
    table.add_row("Всего элементов", str(summary['total_elements']))
    table.add_row("Функций", str(len(summary['functions'])))
    table.add_row("Классов", str(len(summary['classes'])))
    table.add_row("Импортов", str(len(summary['imports'])))
    
    console.print(table)
    
    # Детали функций
    if summary['functions']:
        console.print("\n[bold yellow]Функции:[/bold yellow]")
        for func in summary['functions'][:10]:  # Показываем первые 10
            metadata = func['metadata']
            console.print(f"• {metadata['name']} (строки {metadata['line_start']}-{metadata['line_end']}, сложность: {metadata['complexity']})")
    
    # Детали классов
    if summary['classes']:
        console.print("\n[bold yellow]Классы:[/bold yellow]")
        for cls in summary['classes']:
            metadata = cls['metadata']
            console.print(f"• {metadata['name']} (строки {metadata['line_start']}-{metadata['line_end']})")

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
    
    console.print("[bold blue]🎯 Интерактивный режим Python RAG System[/bold blue]")
    console.print("Доступные команды: search, analyze, flow, ask, improve, similar, stats, quit")
    
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
            command = Prompt.ask("\n[bold cyan]Команда[/bold cyan]").strip().lower()
            
            if command == 'quit' or command == 'exit':
                console.print("[green]До свидания! 👋[/green]")
                break
            
            elif command == 'search':
                query = Prompt.ask("Поисковый запрос")
                results = rag_engine.search(query, n_results=3)
                
                for i, result in enumerate(results, 1):
                    metadata = result['metadata']
                    console.print(f"\n[bold]{i}. {metadata['name']} ({metadata['type']})[/bold]")
                    console.print(f"Файл: {metadata['file_path']}")
                    console.print(f"Код: {result['document'][:200]}...")
            
            elif command == 'analyze':
                function_name = Prompt.ask("Имя функции для анализа")
                result = analyzer.analyze_function(function_name)
                analyzer.display_analysis_result(result, "function")
            
            elif command == 'flow':
                function_name = Prompt.ask("Имя функции для анализа flow")
                result = analyzer.explain_code_flow(function_name)
                analyzer.display_analysis_result(result, "flow")
            
            elif command == 'ask':
                question = Prompt.ask("Ваш вопрос о коде")
                result = analyzer.answer_code_question(question)
                analyzer.display_analysis_result(result, "question")
            
            elif command == 'improve':
                function_name = Prompt.ask("Имя функции для улучшения")
                result = analyzer.suggest_improvements(function_name)
                analyzer.display_analysis_result(result, "improvements")
            
            elif command == 'similar':
                function_name = Prompt.ask("Имя функции для поиска похожих")
                result = analyzer.find_similar_functions(function_name)
                analyzer.display_analysis_result(result, "similar")
            
            elif command == 'stats':
                stats = rag_engine.get_collection_stats()
                console.print(f"Документов в коллекции: {stats['total_documents']}")
            
            else:
                console.print("[red]Неизвестная команда. Доступные: search, analyze, flow, ask, improve, similar, stats, quit[/red]")
        
        except KeyboardInterrupt:
            console.print("\n[green]До свидания! 👋[/green]")
            break
        except Exception as e:
            console.print(f"[red]Ошибка: {e}[/red]")

if __name__ == '__main__':
    cli() 