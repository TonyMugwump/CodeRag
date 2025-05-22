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
from rich.console import Console

console = Console()

def main():
    """Основная функция демонстрации"""
    
    console.print("[bold blue]🐍 Демонстрация Python RAG System[/bold blue]\n")
    
    # 1. Инициализация RAG движка
    console.print("[yellow]1. Инициализация RAG движка...[/yellow]")
    rag_engine = PythonRAGEngine(
        collection_name="demo_collection",
        persist_directory=".demo_chroma_store"
    )
    
    # 2. Индексация текущего проекта (самой RAG системы)
    console.print("[yellow]2. Индексация проекта...[/yellow]")
    project_path = "./python_rag_system"
    
    if os.path.exists(project_path):
        summary = rag_engine.index_project(project_path)
        
        if summary:
            console.print(f"✅ Проиндексировано {summary['total_elements']} элементов")
            console.print(f"   - Функций: {summary['functions']}")
            console.print(f"   - Классов: {summary['classes']}")
            console.print(f"   - Файлов: {summary['files_analyzed']}")
        else:
            console.print("❌ Ошибка индексации")
            return
    else:
        console.print("❌ Директория проекта не найдена")
        return
    
    # 3. Демонстрация поиска
    console.print("\n[yellow]3. Демонстрация поиска...[/yellow]")
    
    search_queries = [
        "функция для парсинга кода",
        "ChromaDB коллекция",
        "анализ сложности функций"
    ]
    
    for query in search_queries:
        console.print(f"\n🔍 Поиск: '{query}'")
        results = rag_engine.search(query, n_results=2)
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            console.print(f"  {i}. {metadata['name']} ({metadata['type']}) - {metadata['file_path']}")
    
    # 4. Демонстрация LLM анализа (если доступен API ключ)
    console.print("\n[yellow]4. Демонстрация LLM анализа...[/yellow]")
    
    if os.getenv("OPENAI_API_KEY"):
        analyzer = LLMCodeAnalyzer(rag_engine, model="gpt-3.5-turbo")  # Используем более дешевую модель для демо
        
        # Анализируем функцию parse_file
        console.print("\n🧠 Анализ функции 'parse_file':")
        result = analyzer.analyze_function("parse_file")
        
        if "error" not in result:
            console.print("✅ Анализ выполнен успешно")
            # analyzer.display_analysis_result(result, "function")
        else:
            console.print(f"❌ Ошибка анализа: {result['error']}")
        
        # Демонстрация ответа на вопрос
        console.print("\n❓ Ответ на вопрос о коде:")
        question = "Как работает индексация проекта?"
        result = analyzer.answer_code_question(question)
        
        if "error" not in result:
            console.print("✅ Ответ получен")
            # analyzer.display_analysis_result(result, "question")
        else:
            console.print(f"❌ Ошибка: {result['error']}")
    
    else:
        console.print("⚠️  OPENAI_API_KEY не установлен. LLM функции недоступны.")
        console.print("   Установите переменную окружения для полной демонстрации:")
        console.print("   export OPENAI_API_KEY='your-api-key'")
    
    # 5. Статистика коллекции
    console.print("\n[yellow]5. Статистика коллекции...[/yellow]")
    stats = rag_engine.get_collection_stats()
    console.print(f"📊 Документов в коллекции: {stats['total_documents']}")
    console.print(f"📁 Директория: {stats['persist_directory']}")
    console.print(f"🤖 Модель эмбеддингов: {stats['embedding_model']}")
    
    # 6. Генерация документации
    console.print("\n[yellow]6. Генерация документации...[/yellow]")
    docs = rag_engine.generate_project_documentation()
    
    with open("demo_documentation.md", "w", encoding="utf-8") as f:
        f.write(docs)
    
    console.print("📄 Документация сохранена в demo_documentation.md")
    
    console.print("\n[green]🎉 Демонстрация завершена![/green]")
    console.print("\nДля интерактивного использования запустите:")
    console.print("python -m python_rag_system.cli interactive")

def demo_advanced_features():
    """Демонстрация продвинутых возможностей"""
    
    console.print("[bold blue]🚀 Демонстрация продвинутых возможностей[/bold blue]\n")
    
    rag_engine = PythonRAGEngine(collection_name="demo_collection")
    
    # Демонстрация получения контекста функции
    console.print("[yellow]Получение контекста функции...[/yellow]")
    context = rag_engine.get_function_context("parse_file")
    
    if "error" not in context:
        main_func = context['main_function']
        console.print(f"✅ Функция: {main_func['metadata']['name']}")
        console.print(f"   Файл: {main_func['metadata']['file_path']}")
        console.print(f"   Сложность: {main_func['metadata']['complexity']}")
        
        if context.get('callees'):
            console.print(f"   Вызывает: {len(context['callees'])} функций")
        
        if context.get('callers'):
            console.print(f"   Вызывается из: {len(context['callers'])} функций")
    
    # Демонстрация поиска по типу
    console.print("\n[yellow]Поиск только классов...[/yellow]")
    class_results = rag_engine.search("анализ", filter_type="class", n_results=3)
    
    for result in class_results:
        metadata = result['metadata']
        console.print(f"🏗️  {metadata['name']} - {metadata['file_path']}")
    
    # Демонстрация сводки по файлу
    console.print("\n[yellow]Сводка по файлу...[/yellow]")
    file_summary = rag_engine.get_file_summary("python_rag_system/core/rag_engine.py")
    
    if file_summary['total_elements'] > 0:
        console.print(f"📄 Файл содержит:")
        console.print(f"   - Функций: {len(file_summary['functions'])}")
        console.print(f"   - Классов: {len(file_summary['classes'])}")
        console.print(f"   - Импортов: {len(file_summary['imports'])}")

if __name__ == "__main__":
    try:
        main()
        
        # Запускаем продвинутую демонстрацию если есть данные
        if os.path.exists(".demo_chroma_store"):
            console.print("\n" + "="*50)
            demo_advanced_features()
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Демонстрация прервана пользователем[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Ошибка: {e}[/red]")
        import traceback
        traceback.print_exc() 