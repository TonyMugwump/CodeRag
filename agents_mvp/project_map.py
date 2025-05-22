import ast
import os
from collections import defaultdict

def collect_functions_and_calls(project_path):
    function_map = {}  # {func_id: {"file": ..., "doc": ..., "calls": [...], "class": ...}}
    call_graph = defaultdict(list)  # {caller_func_id: [callee_func_id, ...]}

    for root, _, files in os.walk(project_path):
        for fname in files:
            if fname.endswith('.py'):
                path = os.path.join(root, fname)
                with open(path, 'r', encoding='utf-8') as f:
                    source = f.read()
                try:
                    tree = ast.parse(source)
                except Exception:
                    continue

                class_name = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_id = f"{path}:{class_name+'.' if class_name else ''}{node.name}"
                        doc = ast.get_docstring(node)
                        calls = []
                        for n in ast.walk(node):
                            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                                calls.append(n.func.id)
                            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                                calls.append(n.func.attr)
                        function_map[func_id] = {
                            "file": path,
                            "doc": doc,
                            "calls": calls,
                            "class": class_name
                        }
                        for callee in calls:
                            call_graph[func_id].append(callee)
    return function_map, call_graph

def generate_mermaid_call_graph(function_map, call_graph):
    lines = ["graph TD"]
    for caller, callees in call_graph.items():
        caller_label = caller.replace(":", "_").replace(".", "_")
        for callee in callees:
            callee_label = callee.replace(":", "_").replace(".", "_")
            lines.append(f'    {caller_label} --> {callee_label}')
    return "\n".join(lines) 