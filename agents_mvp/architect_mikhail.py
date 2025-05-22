from extract import extract_text_from_file

class ArchitectMikhail:
    def __init__(self, rag_index, code_agent, function_map, call_graph):
        self.rag = rag_index
        self.ilya = code_agent
        self.function_map = function_map
        self.call_graph = call_graph
        self.plan = []

    def select_best_match(self, code_matches):
        # Выбираем функцию с наибольшим количеством входящих связей (кто её вызывает)
        max_incoming = -1
        best = None
        for match in code_matches:
            func_id = match['id']
            incoming = sum(func_id in callees for callees in self.call_graph.values())
            if incoming > max_incoming:
                max_incoming = incoming
                best = match
        return best if best else (code_matches[0] if code_matches else None)

    def process_request(self, user_query: str) -> list:
        task = {
            "type": "modify",
            "what": user_query,
            "how": user_query
        }
        self.plan = [task]
        results = []
        for task in self.plan:
            code_matches = self.rag.query(task['what'])
            print(f"[DEBUG] Найдено {len(code_matches)} кандидатов:")
            for m in code_matches:
                print(f"  - id: {m['id']}, score: {m.get('score')}, meta: {m['meta']}")
            best_match = self.select_best_match(code_matches)
            if best_match:
                print(f"[DEBUG] Выбранный блок: {best_match['id']}, meta: {best_match['meta']}")
                meta = best_match['meta']
                code = extract_text_from_file(meta)
                print(f"[DEBUG] Код, отправляемый Илье (полностью):\n{code}\n")
                new_code = self.ilya.apply_instruction(task['how'], code, meta)
                print(f"[DEBUG] Новый код от LLM (полностью):\n{new_code}\n")
                diff = self.ilya.generate_diff(code, new_code)
                results.append((best_match['id'], diff))
        return results 