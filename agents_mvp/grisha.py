import os
from extract import extract_text_from_file
from rag_index_faiss import azure_get_embeddings  # для совместимости, если нужно
from openai import AzureOpenAI

class GrishaAgent:
    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.client = AzureOpenAI(
            api_version=os.getenv("AZURE_LLM_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),
            api_key=os.getenv("AZURE_LLM_API_KEY"),
        )

    def generate_docstrings_and_update(self, context):
        updated = 0
        no_doc = 0
        for meta in context.code_metas:
            code = self._extract_code(meta)
            if not code:
                print(f"[Grisha] Не удалось извлечь код для {meta.get('filepath', meta)}")
                continue
            # Проверяем, есть ли docstring
            if meta.get("docstring") or meta.get("doc"):
                print(f"[Grisha] Пропускаю {meta.get('filepath', meta)} — docstring уже есть.")
                continue
            no_doc += 1
            print(f"[Grisha] Обрабатываю {meta.get('filepath', meta)} (docstring отсутствует)")
            docstring = self.ask_llm_for_docstring(code, meta)
            print(f"[Grisha] LLM docstring для {meta.get('filepath', meta)}: {repr(docstring)[:120]}")
            if docstring:
                self.insert_docstring_in_file(meta["filepath"], meta, docstring)
                print(f"[Grisha] Docstring добавлен в {meta['filepath']}")
                updated += 1
            else:
                print(f"[Grisha] Не удалось получить docstring для {meta.get('filepath', meta)}")
        print(f"[Grisha] Всего функций/классов без docstring: {no_doc}")
        context.update()
        flow_count = self._count_flows_in_project(context)
        return f"Обновлено {updated} функций/классов, найдено {flow_count} flows."

    def _extract_code(self, func_info):
        try:
            return extract_text_from_file(func_info)
        except Exception:
            return None

    def ask_llm_for_docstring(self, code, meta):
        prompt = f"""
Ты — эксперт по Python и документации. Сгенерируй подробный docstring для следующей функции или класса. В docstring обязательно:
- Опиши назначение и бизнес-логику.
- Укажи, к какому flow/сценарию относится объект (если понятно).
- Опиши входные и выходные параметры.
- Приведи пример использования (если уместно).
- Используй стиль Google или NumPy.

Код:
{code}
"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Ты — эксперт по Python и документации."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=2048,
                model=self.llm_model
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Grisha] Ошибка запроса к LLM: {e}")
            return None

    def insert_docstring_in_file(self, filepath, func_info, docstring):
        # Вставляет docstring в исходный код (упрощённо: только если docstring отсутствует)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        start = func_info["start_line"] - 1
        end = func_info["end_line"]
        # Проверяем, есть ли уже docstring
        if '"""' in ''.join(lines[start:end]) or "'''" in ''.join(lines[start:end]):
            return  # docstring уже есть
        indent = ' ' * (len(lines[start]) - len(lines[start].lstrip()))
        docstring_lines = [f'{indent}"""{docstring}"""\n']
        lines.insert(start + 1, ''.join(docstring_lines))
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    def _count_flows_in_project(self, context):
        count = 0
        for func in context.function_map.values():
            if func.get("doc") and ("flow" in func["doc"].lower() or "pipeline" in func["doc"].lower() or "scenario" in func["doc"].lower()):
                count += 1
        return count 