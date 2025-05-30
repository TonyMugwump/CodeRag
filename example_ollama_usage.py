#!/usr/bin/env python3
"""
Пример использования Python RAG System с Ollama эмбеддингами
"""

import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from python_rag_system.core.rag_engine import PythonRAGEngine
from python_rag_system.core.llm_integration import LLMCodeAnalyzer
from python_rag_system.core import lang

console = Console()

def main():
    console.print(Panel.fit(lang.tr("demo_title"), style="bold blue"))
    
    # Конфигурация для Ollama
    embedding_config = {
        'embedding_provider': 'ollama',
        'embedding_model': 'nomic-embed-text:v1.5',
        'ollama_base_url': 'http://localhost:11434'
    }
    
    try:
        console.print(lang.tr("initializing_rag"))
        
        # Создаем RAG движок с Ollama эмбеддингами
        rag_engine = PythonRAGEngine(
            collection_name="ollama_demo_collection_v2",  # Новое имя коллекции
            persist_directory=".chroma_ollama_store",
            **embedding_config
        )
        
        console.print(lang.tr("rag_initialized"))
        
        # Показываем статистику
        stats = rag_engine.get_collection_stats()
        
        table = Table(title=lang.tr("system_config"))
        table.add_column(lang.tr("parameter"), style="cyan")
        table.add_column(lang.tr("value"), style="green")
        
        table.add_row(lang.tr("embedding_provider"), stats['embedding_provider'])
        table.add_row(lang.tr("embedding_model"), stats['embedding_model'])
        table.add_row(lang.tr("embedding_dimension"), str(stats['embedding_dimension']))
        table.add_row(lang.tr("documents_in_collection"), str(stats['total_documents']))
        
        console.print(table)
        
        # Если коллекция пустая, индексируем проект
        if stats['total_documents'] == 0:
            console.print(lang.tr("indexing_project"))
            project_path = "./python_rag_system"
            
            summary = rag_engine.index_project(project_path)
            
            if summary:
                console.print(lang.tr("indexing_complete", total_elements=summary['total_elements']))
                console.print(lang.tr('indexing_complete', file='demo_documentation.md'))
            else:
                console.print(lang.tr("indexing_error"))
                return
        else:
            console.print(lang.tr("using_existing_index"))
        
        # Демонстрация поиска
        console.print(lang.tr("semantic_search_demo"))
        
        search_queries = [
            lang.tr("query_parse_python_code"),
            lang.tr("query_create_embeddings"),
            lang.tr("query_work_with_chromadb"),
            lang.tr("query_code_complexity_analysis")
        ]
        
        for query in search_queries:
            console.print(lang.tr("search_query", query=query))
            
            results = rag_engine.search(query, n_results=3)
            
            if results:
                for i, result in enumerate(results[:2], 1):
                    metadata = result['metadata']
                    distance = result.get('distance', 0)
                    relevance = 1 - distance
                    
                    console.print(lang.tr("search_result", index=i, name=metadata['name'], type=metadata['type'], relevance=relevance))
                    console.print(lang.tr("file_location", file_path=metadata['file_path'], line_start=metadata['line_start']))
            else:
                console.print(lang.tr("no_results_found"))
        
        # Демонстрация анализа функций
        console.print(lang.tr("function_analysis_demo"))
        
        # Ищем интересную функцию для анализа
        function_results = rag_engine.search("parse_file", filter_type="function", n_results=1)
        
        if function_results:
            function_name = function_results[0]['metadata']['name']
            console.print(lang.tr("analyzing_function", function_name=function_name))
            
            # Получаем контекст функции
            context = rag_engine.get_function_context(function_name)
            
            if 'error' not in context:
                console.print(lang.tr("function_complexity", complexity=context['main_function']['metadata']['complexity']))
                
                if context.get('flow_info'):
                    flow = context['flow_info']
                    if flow.get('calls'):
                        console.print(lang.tr("function_calls", count=len(flow['calls'])))
                    if flow.get('called_by'):
                        console.print(lang.tr("function_called_by", count=len(flow['called_by'])))
        
        # Демонстрация сравнения с SentenceTransformers
        console.print(lang.tr("comparison_with_st_demo"))
        
        # Создаем второй движок с SentenceTransformers для сравнения
        st_engine = PythonRAGEngine(
            collection_name="st_demo_collection",
            persist_directory=".chroma_st_store",
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Сравниваем результаты поиска
        test_query = lang.tr("query_create_embeddings")
        
        console.print(lang.tr("comparison_query", query=test_query))
        
        ollama_results = rag_engine.search(test_query, n_results=3)
        st_results = st_engine.search(test_query, n_results=3)
        
        comparison_table = Table(title=lang.tr("search_results_comparison"))
        comparison_table.add_column(lang.tr("position"), style="cyan")
        comparison_table.add_column(lang.tr("ollama_results"), style="green")
        comparison_table.add_column(lang.tr("st_results"), style="yellow")
        
        max_results = max(len(ollama_results), len(st_results))
        
        for i in range(min(3, max_results)):
            ollama_name = ollama_results[i]['metadata']['name'] if i < len(ollama_results) else lang.tr("not_available")
            st_name = st_results[i]['metadata']['name'] if i < len(st_results) else lang.tr("not_available")
            
            ollama_rel = f"{1-ollama_results[i].get('distance', 1):.3f}" if i < len(ollama_results) else lang.tr("not_available")
            st_rel = f"{1-st_results[i].get('distance', 1):.3f}" if i < len(st_results) else lang.tr("not_available")
            
            comparison_table.add_row(
                str(i+1),
                f"{ollama_name} ({ollama_rel})",
                f"{st_name} ({st_rel})"
            )
        
        console.print(comparison_table)
        
        console.print(lang.tr("demo_complete"))
        console.print("\n[green]🎉 " + lang.tr('analysis_done') + "[/green]")
        console.print("\n[blue]" + lang.tr('demo_interactive') + "[/blue]")
        console.print("python -m python_rag_system.cli interactive")
        
        # Рекомендации
        recommendations = Panel(
            lang.tr("recommendations"),
            title=lang.tr("tips"),
            border_style="blue"
        )
        
        console.print(recommendations)
        
    except Exception as e:
        console.print(f"[red]{lang.tr('error', error=e)}[/red]")
        console.print("\n[yellow]" + lang.tr('set_env') + "[/yellow]")
        console.print(lang.tr("ensure_conditions"))

if __name__ == "__main__":
    main()