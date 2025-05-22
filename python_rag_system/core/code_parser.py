import ast
import os
import inspect
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import networkx as nx
from rich.console import Console
from rich.tree import Tree

console = Console()

@dataclass
class CodeElement:
    """Базовый класс для элементов кода"""
    name: str
    type: str  # 'function', 'class', 'method', 'variable', 'import'
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    docstring: Optional[str] = None
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    calls: Set[str] = field(default_factory=set)
    complexity: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type,
            'file_path': self.file_path,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'source_code': self.source_code,
            'docstring': self.docstring,
            'parent': self.parent,
            'children': list(self.children),
            'dependencies': list(self.dependencies),
            'calls': list(self.calls),
            'complexity': self.complexity
        }

class PythonCodeParser:
    """Парсер Python кода для извлечения структуры и зависимостей"""
    
    def __init__(self):
        self.elements: Dict[str, CodeElement] = {}
        self.call_graph = nx.DiGraph()
        self.dependency_graph = nx.DiGraph()
        self.file_tree = Tree("Project Structure")
        
    def parse_file(self, file_path: str) -> List[CodeElement]:
        """Парсит один Python файл"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            elements = []
            
            # Извлекаем все элементы из файла
            for node in ast.walk(tree):
                element = self._extract_element(node, file_path, source)
                if element:
                    elements.append(element)
                    self.elements[f"{file_path}::{element.name}"] = element
            
            # Строим граф вызовов для этого файла
            self._build_call_graph(tree, file_path)
            
            return elements
            
        except Exception as e:
            console.print(f"[red]Ошибка парсинга {file_path}: {e}[/red]")
            return []
    
    def parse_project(self, project_path: str, exclude_patterns: List[str] = None) -> Dict[str, List[CodeElement]]:
        """Парсит весь проект"""
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', '.venv', 'venv', 'node_modules', '.pytest_cache']
        
        project_elements = {}
        
        for root, dirs, files in os.walk(project_path):
            # Исключаем ненужные директории
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, project_path)
                    
                    elements = self.parse_file(file_path)
                    if elements:
                        project_elements[relative_path] = elements
        
        # Строим общие графы зависимостей
        self._build_dependency_graph()
        self._calculate_complexity()
        
        return project_elements
    
    def _extract_element(self, node: ast.AST, file_path: str, source: str) -> Optional[CodeElement]:
        """Извлекает элемент кода из AST узла"""
        if isinstance(node, ast.FunctionDef):
            return self._extract_function(node, file_path, source)
        elif isinstance(node, ast.AsyncFunctionDef):
            return self._extract_function(node, file_path, source, is_async=True)
        elif isinstance(node, ast.ClassDef):
            return self._extract_class(node, file_path, source)
        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            return self._extract_import(node, file_path, source)
        
        return None
    
    def _extract_function(self, node: ast.FunctionDef, file_path: str, source: str, is_async: bool = False) -> CodeElement:
        """Извлекает функцию из AST"""
        lines = source.split('\n')
        source_code = '\n'.join(lines[node.lineno-1:node.end_lineno])
        
        # Извлекаем docstring
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        # Находим вызовы функций
        calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr)
        
        func_type = 'async_function' if is_async else 'function'
        
        return CodeElement(
            name=node.name,
            type=func_type,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            source_code=source_code,
            docstring=docstring,
            calls=calls
        )
    
    def _extract_class(self, node: ast.ClassDef, file_path: str, source: str) -> CodeElement:
        """Извлекает класс из AST"""
        lines = source.split('\n')
        source_code = '\n'.join(lines[node.lineno-1:node.end_lineno])
        
        # Извлекаем docstring
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        # Находим методы класса
        methods = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(child.name)
        
        return CodeElement(
            name=node.name,
            type='class',
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            source_code=source_code,
            docstring=docstring,
            children=methods
        )
    
    def _extract_import(self, node: ast.AST, file_path: str, source: str) -> CodeElement:
        """Извлекает импорт из AST"""
        lines = source.split('\n')
        source_code = lines[node.lineno-1] if node.lineno <= len(lines) else ""
        
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            import_name = ', '.join(names)
        else:  # ast.ImportFrom
            module = node.module or ''
            names = [alias.name for alias in node.names]
            import_name = f"from {module} import {', '.join(names)}"
        
        return CodeElement(
            name=import_name,
            type='import',
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.lineno,
            source_code=source_code
        )
    
    def _build_call_graph(self, tree: ast.AST, file_path: str):
        """Строит граф вызовов для файла"""
        current_function = None
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                current_function = f"{file_path}::{node.name}"
                self.call_graph.add_node(current_function)
            
            elif isinstance(node, ast.Call) and current_function:
                called_func = None
                if isinstance(node.func, ast.Name):
                    called_func = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    called_func = node.func.attr
                
                if called_func:
                    called_full = f"{file_path}::{called_func}"
                    self.call_graph.add_edge(current_function, called_full)
    
    def _build_dependency_graph(self):
        """Строит граф зависимостей между модулями"""
        for element_key, element in self.elements.items():
            if element.type == 'import':
                file_path = element.file_path
                # Добавляем зависимость файла от импортируемого модуля
                self.dependency_graph.add_edge(file_path, element.name)
    
    def _calculate_complexity(self):
        """Вычисляет сложность для каждого элемента"""
        for element in self.elements.values():
            if element.type in ['function', 'async_function']:
                # Простая метрика сложности: количество строк + количество вызовов
                lines_count = element.line_end - element.line_start + 1
                calls_count = len(element.calls)
                element.complexity = lines_count + calls_count * 2
    
    def get_function_flow(self, function_name: str) -> Dict[str, Any]:
        """Получает flow для конкретной функции"""
        matching_functions = [key for key in self.elements.keys() if function_name in key]
        
        if not matching_functions:
            return {}
        
        func_key = matching_functions[0]
        element = self.elements[func_key]
        
        # Проверяем, что элемент является функцией
        if element.type not in ['function', 'async_function']:
            return {
                'element': element.to_dict(),
                'calls': [],
                'called_by': [],
                'complexity': element.complexity
            }
        
        # Находим ключ функции в графе вызовов (формат: file_path::function_name)
        call_graph_key = f"{element.file_path}::{element.name}"
        
        called_by = []
        calls_this = []
        
        # Проверяем, что функция есть в графе вызовов перед попыткой получить связи
        if self.call_graph.has_node(call_graph_key):
            # Находим все функции, которые вызывает данная функция
            called_by = list(self.call_graph.successors(call_graph_key))
            # Находим все функции, которые вызывают данную функцию
            calls_this = list(self.call_graph.predecessors(call_graph_key))
        
        return {
            'element': element.to_dict(),
            'calls': called_by,
            'called_by': calls_this,
            'complexity': element.complexity
        }
    
    def export_call_graph_mermaid(self) -> str:
        """Экспортирует граф вызовов в формат Mermaid"""
        mermaid = ["graph TD"]
        
        for edge in self.call_graph.edges():
            source = edge[0].split("::")[-1]
            target = edge[1].split("::")[-1]
            mermaid.append(f"    {source} --> {target}")
        
        return "\n".join(mermaid)
    
    def get_project_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по проекту"""
        functions_count = len([e for e in self.elements.values() if e.type in ['function', 'async_function']])
        classes_count = len([e for e in self.elements.values() if e.type == 'class'])
        imports_count = len([e for e in self.elements.values() if e.type == 'import'])
        
        return {
            'total_elements': len(self.elements),
            'functions': functions_count,
            'classes': classes_count,
            'imports': imports_count,
            'files_analyzed': len(set(e.file_path for e in self.elements.values())),
            'call_graph_edges': self.call_graph.number_of_edges(),
            'dependency_graph_edges': self.dependency_graph.number_of_edges()
        } 