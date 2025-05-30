#!/usr/bin/env python3
"""
Пример использования Python RAG System

Этот скрипт демонстрирует основные возможности системы:
1. Индексация проекта
2. Поиск по коду
3. Анализ функций с помощью LLM
4. Построение flow диаграмм
"""

import os
from python_rag_system import PythonRAGEngine, LLMCodeAnalyzer
from python_rag_system.core import lang
from rich.console import Console

console = Console()

def main():
    """Основная функция демонстрации"""
    
    console.print(lang.tr("demo_start"))
    
    # 1. Инициализация RAG движка
    console.print(lang.tr("initializing_rag_engine"))
    rag_engine = PythonRAGEngine(
        collection_name="demo_collection",
        persist_directory=".demo_chroma_store"
    )
    
    # 2. Индексация текущего проекта (самой RAG системы)
    console.print(lang.tr("indexing_project"))
    project_path = "./python_rag_system"
    
    if os.path.exists(project_path):
        summary = rag_engine.index_project(project_path)
        
        if summary:
            console.print(lang.tr("indexing_complete", total_elements=summary['total_elements']))
            console.print(lang.tr("indexing_details", functions=summary['functions'], classes=summary['classes'], files_analyzed=summary['files_analyzed']))
        else:
            console.print(lang.tr("indexing_error"))
            return
    else:
        console.print(lang.tr("project_directory_not_found"))
        return
    
    # 3. Демонстрация поиска
    console.print(lang.tr("search_demo"))
    
    search_queries = [
        lang.tr("search_query_parse_code"),
        lang.tr("search_query_chromadb_collection"),
        lang.tr("search_query_function_complexity_analysis")
    ]
    
    for query in search_queries:
        console.print(lang.tr("search_query", query=query))
        results = rag_engine.search(query, n_results=2)
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            console.print(lang.tr("search_result", index=i, name=metadata['name'], type=metadata['type'], file_path=metadata['file_path']))
    
    # 4. Демонстрация LLM анализа (если доступен API ключ)
    console.print(lang.tr("llm_analysis_demo"))
    
    if os.getenv("OPENAI_API_KEY"):
        analyzer = LLMCodeAnalyzer(rag_engine, model="gpt-3.5-turbo")  # Используем более дешевую модель для демо
        
        # Анализируем функцию parse_file
        console.print(lang.tr("analyzing_function", function_name="parse_file"))
        result = analyzer.analyze_function("parse_file")
        
        if "error" not in result:
            console.print(lang.tr("analysis_success"))
            # analyzer.display_analysis_result(result, "function")
        else:
            console.print(lang.tr("analysis_error", error=result['error']))
        
        # Демонстрация ответа на вопрос
        console.print(lang.tr("answering_question"))
        question = lang.tr("question_how_project_indexing_works")
        result = analyzer.answer_code_question(question)
        
        if "error" not in result:
            console.print(lang.tr("answer_success"))
            # analyzer.display_analysis_result(result, "question")
        else:
            console.print(lang.tr("answer_error", error=result['error']))
    
    else:
        console.print(lang.tr('llm_unavailable'))
        console.print(lang.tr('set_env'))
    
    # 5. Статистика коллекции
    console.print(lang.tr("collection_stats"))
    stats = rag_engine.get_collection_stats()
    console.print(lang.tr("collection_documents", total_documents=stats['total_documents']))
    console.print(lang.tr("collection_directory", directory=stats['persist_directory']))
    console.print(lang.tr("collection_embedding_model", model=stats['embedding_model']))
    
    # 6. Генерация документации
    console.print(lang.tr("generating_documentation"))
    docs = rag_engine.generate_project_documentation()
    
    with open("demo_documentation.md", "w", encoding="utf-8") as f:
        f.write(docs)
    
    console.print(lang.tr('indexing_complete', file='demo_documentation.md'))
    console.print("\n[green]🎉 " + lang.tr('analysis_done') + "[/green]")
    console.print("\n" + lang.tr('demo_interactive'))
    console.print("python -m python_rag_system.cli interactive")
    
    console.print(lang.tr("demo_complete"))
    console.print(lang.tr("interactive_usage_instruction"))

def demo_advanced_features():
    """Демонстрация продвинутых возможностей"""
    
    console.print(lang.tr("advanced_features_demo"))
    
    rag_engine = PythonRAGEngine(collection_name="demo_collection")
    
    # Демонстрация получения контекста функции
    console.print(lang.tr("getting_function_context"))
    context = rag_engine.get_function_context("parse_file")
    
    if "error" not in context:
        main_func = context['main_function']
        console.print(lang.tr("function_context", name=main_func['metadata']['name'], file_path=main_func['metadata']['file_path'], complexity=main_func['metadata']['complexity']))
        
        if context.get('callees'):
            console.print(lang.tr("function_callees", count=len(context['callees'])))
        
        if context.get('callers'):
            console.print(lang.tr("function_callers", count=len(context['callers'])))
    
    # Демонстрация поиска по типу
    console.print(lang.tr("search_classes_only"))
    class_results = rag_engine.search(lang.tr("search_query_analysis"), filter_type="class", n_results=3)
    
    for result in class_results:
        metadata = result['metadata']
        console.print(lang.tr("class_result", name=metadata['name'], file_path=metadata['file_path']))
    
    # Демонстрация сводки по файлу
    console.print(lang.tr("file_summary"))
    file_summary = rag_engine.get_file_summary("python_rag_system/core/rag_engine.py")
    
    if file_summary['total_elements'] > 0:
        console.print(lang.tr("file_contains"))
        console.print(lang.tr("file_functions", count=len(file_summary['functions'])))
        console.print(lang.tr("file_classes", count=len(file_summary['classes'])))
        console.print(lang.tr("file_imports", count=len(file_summary['imports'])))

if __name__ == "__main__":
    try:
        main()
        
        # Запускаем продвинутую демонстрацию если есть данные
        if os.path.exists(".demo_chroma_store"):
            console.print("\n" + "="*50)
            demo_advanced_features()
            
    except KeyboardInterrupt:
        console.print("\n[yellow]" + lang.tr('interrupted') + "[/yellow]")
    except Exception as e:
        console.print(f"\n[red]{lang.tr('error', error=e)}[/red]")
        import traceback
        traceback.print_exc()