"""
Python RAG System - Система анализа кода с ChromaDB и LLM

Этот пакет предоставляет инструменты для:
- Парсинга Python кода и извлечения структуры
- Индексации кода в ChromaDB
- Построения графов вызовов и зависимостей
- Интеллектуального поиска и анализа с помощью LLM
"""

from .core.code_parser import PythonCodeParser, CodeElement
from .core.rag_engine import PythonRAGEngine
from .core.llm_integration import LLMCodeAnalyzer

__version__ = "1.0.0"
__author__ = "Python RAG System"
__email__ = "contact@pythonrag.com"

__all__ = [
    "PythonCodeParser",
    "CodeElement", 
    "PythonRAGEngine",
    "LLMCodeAnalyzer"
] 