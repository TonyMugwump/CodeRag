import os
import json
import subprocess
import pytest

JS_PROJECT_PATH = "conan_gemini_repo"
JS_INDEX_OUTPUT = "js_index_test_output.json"

def run_js_parser(files, output_file):
    cmd = ["node", "js_parser.js"] + files
    with open(output_file, "w") as out:
        subprocess.run(cmd, stdout=out, check=True)

def get_all_js_files():
    js_files = []
    for root, _, files in os.walk(JS_PROJECT_PATH):
        for f in files:
            if f.endswith(".js"):
                js_files.append(os.path.join(root, f))
    return js_files

def test_js_parser_batch():
    files = get_all_js_files()
    run_js_parser(files, JS_INDEX_OUTPUT)
    with open(JS_INDEX_OUTPUT) as f:
        data = json.load(f)
    assert "elements" in data and isinstance(data["elements"], list)
    assert "callEdges" in data and isinstance(data["callEdges"], list)
    assert len(data["elements"]) > 0
    assert len(data["callEdges"]) > 0

def test_js_parser_function_names():
    files = get_all_js_files()
    run_js_parser(files, JS_INDEX_OUTPUT)
    with open(JS_INDEX_OUTPUT) as f:
        data = json.load(f)
    function_names = [el["name"] for el in data["elements"] if el["type"] in ("function", "function_expression", "arrow_function")]
    assert "getUserInfo" in function_names
    assert "registerUser" in function_names
    assert any(fn for fn in function_names if fn != "<anonymous>")

def test_js_parser_imports():
    files = get_all_js_files()
    run_js_parser(files, JS_INDEX_OUTPUT)
    with open(JS_INDEX_OUTPUT) as f:
        data = json.load(f)
    import_names = [el["name"] for el in data["elements"] if el["type"] == "import"]
    assert any("telegraf" in name for name in import_names)
    assert any("dotenv" in name for name in import_names)

def test_js_parser_call_graph():
    files = get_all_js_files()
    run_js_parser(files, JS_INDEX_OUTPUT)
    with open(JS_INDEX_OUTPUT) as f:
        data = json.load(f)
    call_edges = data["callEdges"]
    assert any(edge["to"]["name"] == "getUserInfo" for edge in call_edges)
    assert any(edge["from"]["name"] == "registerUser" for edge in call_edges) or any(edge["from"]["name"] == "getUserInfo" for edge in call_edges)

def test_js_parser_arrow_functions():
    files = get_all_js_files()
    run_js_parser(files, JS_INDEX_OUTPUT)
    with open(JS_INDEX_OUTPUT) as f:
        data = json.load(f)
    arrow_funcs = [el for el in data["elements"] if el["type"] == "arrow_function"]
    assert len(arrow_funcs) > 0
    assert any(el["name"] == "<anonymous>" for el in arrow_funcs)

if __name__ == "__main__":
    pytest.main([__file__])
