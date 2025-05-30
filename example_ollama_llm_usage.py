#!/usr/bin/env python3
"""
Пример использования Python RAG System с Ollama LLM
"""

import os
import sys
from pathlib import Path

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent))

from python_rag_system.core.rag_engine import PythonRAGEngine
from python_rag_system.core.unified_llm_analyzer import UnifiedLLMCodeAnalyzer
from python_rag_system.core import lang
from rich.console import Console

console = Console()

def main():
    console.print(lang.tr("example_usage"))
    
    # Настройки
    project_path = "./python_rag_system"  # Анализируем саму систему
    collection_name = "ollama_test_rag"
    persist_dir = ".chroma_ollama_test"
    
    try:
        # 1. Создаем RAG движок с Ollama эмбеддингами
        console.print(lang.tr("initializing_rag_engine"))
        rag_engine = PythonRAGEngine(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_provider="ollama",
            embedding_model="nomic-embed-text:v1.5",
            ollama_base_url="http://localhost:11434"
        )
        
        # 2. Индексируем проект
        console.print(lang.tr("indexing_project"))
        summary = rag_engine.index_project(project_path)
        
        if summary:
            console.print(lang.tr("indexing_complete", total_elements=summary['total_elements']))
        else:
            console.print(lang.tr("indexing_error"))
            return
        
        # 3. Создаем LLM анализатор с Ollama
        console.print(lang.tr("initializing_llm_analyzer"))
        analyzer = UnifiedLLMCodeAnalyzer(
            rag_engine=rag_engine,
            llm_provider="ollama",
            model="gemma3:27b",  # Используем gemma3:27b
            ollama_base_url="http://localhost:11434"
        )
        
        if not analyzer.available:
            console.print(lang.tr("ollama_unavailable"))
            return
        
        # 4. Тестируем поиск
        console.print(lang.tr("testing_semantic_search"))
        search_results = rag_engine.search(lang.tr("search_query"), n_results=3)
        
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            console.print(lang.tr("search_result", index=i, name=metadata['name'], type=metadata['type'], file_path=metadata['file_path']))
        
        # 5. Анализируем функцию с помощью LLM
        if search_results:
            console.print(lang.tr("analyzing_function"))
            function_name = search_results[0]['metadata']['name']
            
            console.print(lang.tr("analyzing_function_name", function_name=function_name))
            result = analyzer.analyze_function(function_name)
            analyzer.display_analysis_result(result, "function")
        
        # 6. Задаем вопрос о коде
        console.print(lang.tr("asking_code_question"))
        question = lang.tr("code_question")
        result = analyzer.answer_code_question(question)
        analyzer.display_analysis_result(result, "question")
        
        # 7. Анализируем flow выполнения
        console.print(lang.tr("analyzing_execution_flow"))
        if search_results:
            function_name = search_results[0]['metadata']['name']
            result = analyzer.explain_code_flow(function_name)
            analyzer.display_analysis_result(result, "flow")
        
        # 8. Предлагаем улучшения
        console.print(lang.tr("suggesting_improvements"))
        if search_results:
            function_name = search_results[0]['metadata']['name']
            result = analyzer.suggest_improvements(function_name)
            analyzer.display_analysis_result(result, "improvements")
        
        console.print("\n[green]🎉 " + lang.tr('analysis_done') + "[/green]")
        console.print("\n[blue]" + lang.tr('demo_interactive') + "[/blue]")
        console.print("python -m python_rag_system.cli interactive")
        
    except Exception as e:
        console.print(f"[red]{lang.tr('error', error=e)}[/red]")
        console.print("\n[yellow]" + lang.tr('set_env') + "[/yellow]")

if __name__ == "__main__":
    main()