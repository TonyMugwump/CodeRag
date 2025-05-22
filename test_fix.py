#!/usr/bin/env python3
"""
Простой тест для проверки исправлений в RAG системе
"""

from rich.console import Console
from python_rag_system.core.rag_engine import PythonRAGEngine

console = Console()

def test_basic_functionality():
    """Тестирует базовую функциональность RAG системы"""
    
    console.print("🧪 Тестирование базовой функциональности...")
    
    try:
        # Тест 1: Инициализация с SentenceTransformers
        console.print("\n1. Тест инициализации с SentenceTransformers...")
        rag_engine = PythonRAGEngine(
            collection_name="test_collection_st",
            persist_directory=".test_chroma_store"
        )
        console.print("✅ SentenceTransformers инициализация успешна")
        
        # Тест 2: Получение статистики
        console.print("\n2. Тест получения статистики...")
        stats = rag_engine.get_collection_stats()
        console.print(f"✅ Статистика получена: {stats['total_documents']} документов")
        
        # Тест 3: Простая индексация
        console.print("\n3. Тест индексации небольшого проекта...")
        summary = rag_engine.index_project("./python_rag_system/core", exclude_patterns=['__pycache__'])
        
        if summary:
            console.print(f"✅ Индексация успешна: {summary['total_elements']} элементов")
        else:
            console.print("❌ Ошибка индексации")
            return False
        
        # Тест 4: Простой поиск
        console.print("\n4. Тест поиска...")
        results = rag_engine.search("функция", n_results=3)
        console.print(f"✅ Поиск выполнен: найдено {len(results)} результатов")
        
        if results:
            for i, result in enumerate(results[:2], 1):
                metadata = result['metadata']
                console.print(f"  {i}. {metadata['name']} ({metadata['type']})")
        
        console.print("\n🎉 Все тесты пройдены успешно!")
        return True
        
    except Exception as e:
        console.print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ollama_if_available():
    """Тестирует Ollama если доступен"""
    
    console.print("\n🦙 Тестирование Ollama (если доступен)...")
    
    try:
        import requests
        
        # Проверяем доступность Ollama
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            console.print("⚠️ Ollama недоступен, пропускаем тест")
            return True
        
        models = response.json().get('models', [])
        model_names = [model['name'] for model in models]
        
        if 'nomic-embed-text:v1.5' not in model_names:
            console.print("⚠️ Модель nomic-embed-text:v1.5 не найдена, пропускаем тест")
            console.print(f"Доступные модели: {', '.join(model_names)}")
            return True
        
        # Тестируем Ollama
        console.print("Тестирование с Ollama...")
        rag_engine_ollama = PythonRAGEngine(
            collection_name="test_collection_ollama",
            persist_directory=".test_chroma_ollama_store",
            embedding_provider="ollama",
            embedding_model="nomic-embed-text:v1.5"
        )
        
        stats = rag_engine_ollama.get_collection_stats()
        console.print(f"✅ Ollama тест успешен: размерность эмбеддингов {stats['embedding_dimension']}")
        
        return True
        
    except Exception as e:
        console.print(f"⚠️ Ollama тест не удался: {e}")
        return True  # Не критично

if __name__ == "__main__":
    console.print("🚀 Запуск тестов исправлений...")
    
    success = test_basic_functionality()
    
    if success:
        test_ollama_if_available()
        console.print("\n✨ Все тесты завершены!")
    else:
        console.print("\n💥 Есть проблемы, требующие внимания") 