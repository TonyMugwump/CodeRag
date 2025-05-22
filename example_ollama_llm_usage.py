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
from rich.console import Console

console = Console()

def main():
    console.print("[bold blue]🚀 Пример использования Python RAG System с Ollama LLM[/bold blue]")
    
    # Настройки
    project_path = "./python_rag_system"  # Анализируем саму систему
    collection_name = "ollama_test_rag"
    persist_dir = ".chroma_ollama_test"
    
    try:
        # 1. Создаем RAG движок с Ollama эмбеддингами
        console.print("\n[yellow]1. Инициализация RAG движка с Ollama эмбеддингами...[/yellow]")
        rag_engine = PythonRAGEngine(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_provider="ollama",
            embedding_model="nomic-embed-text:v1.5",
            ollama_base_url="http://localhost:11434"
        )
        
        # 2. Индексируем проект
        console.print("\n[yellow]2. Индексация проекта...[/yellow]")
        summary = rag_engine.index_project(project_path)
        
        if summary:
            console.print(f"[green]✅ Проиндексировано {summary['total_elements']} элементов[/green]")
        else:
            console.print("[red]❌ Ошибка индексации[/red]")
            return
        
        # 3. Создаем LLM анализатор с Ollama
        console.print("\n[yellow]3. Инициализация Ollama LLM анализатора...[/yellow]")
        analyzer = UnifiedLLMCodeAnalyzer(
            rag_engine=rag_engine,
            llm_provider="ollama",
            model="gemma3:27b",  # Используем gemma3:27b
            ollama_base_url="http://localhost:11434"
        )
        
        if not analyzer.available:
            console.print("[red]❌ Ollama LLM недоступен. Убедитесь что:[/red]")
            console.print("[red]  - Ollama запущен (ollama serve)[/red]")
            console.print("[red]  - Модель gemma3:27b установлена (ollama pull gemma3:27b)[/red]")
            return
        
        # 4. Тестируем поиск
        console.print("\n[yellow]4. Тестирование семантического поиска...[/yellow]")
        search_results = rag_engine.search("функция для анализа кода", n_results=3)
        
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            console.print(f"[cyan]{i}. {metadata['name']} ({metadata['type']}) - {metadata['file_path']}[/cyan]")
        
        # 5. Анализируем функцию с помощью LLM
        if search_results:
            console.print("\n[yellow]5. Анализ функции с помощью Ollama LLM...[/yellow]")
            function_name = search_results[0]['metadata']['name']
            
            console.print(f"[cyan]Анализируем функцию: {function_name}[/cyan]")
            result = analyzer.analyze_function(function_name)
            analyzer.display_analysis_result(result, "function")
        
        # 6. Задаем вопрос о коде
        console.print("\n[yellow]6. Задаем вопрос о коде...[/yellow]")
        question = "Как работает индексация проекта в этой системе?"
        result = analyzer.answer_code_question(question)
        analyzer.display_analysis_result(result, "question")
        
        # 7. Анализируем flow выполнения
        console.print("\n[yellow]7. Анализ flow выполнения...[/yellow]")
        if search_results:
            function_name = search_results[0]['metadata']['name']
            result = analyzer.explain_code_flow(function_name)
            analyzer.display_analysis_result(result, "flow")
        
        # 8. Предлагаем улучшения
        console.print("\n[yellow]8. Предложения по улучшению...[/yellow]")
        if search_results:
            function_name = search_results[0]['metadata']['name']
            result = analyzer.suggest_improvements(function_name)
            analyzer.display_analysis_result(result, "improvements")
        
        console.print("\n[green]🎉 Демонстрация завершена успешно![/green]")
        console.print("\n[blue]Теперь вы можете использовать CLI команды:[/blue]")
        console.print("[blue]python -m python_rag_system.cli analyze <function_name> --llm-provider ollama --model gemma3:27b[/blue]")
        console.print("[blue]python -m python_rag_system.cli interactive --llm-provider ollama --llm-model gemma3:27b[/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Ошибка: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

if __name__ == "__main__":
    main() 