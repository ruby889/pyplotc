"""Shared replacement and for-loop expansion for structure file parsers."""

import re

FOR_LOOP_START_RE = re.compile(
    r'for\s*\(\s*(?:int\s+)?(\w+)\s*=\s*(\d+)\s*;\s*\1\s*(<=|<|>=|>)\s*([^;]+?)\s*;'
)


def apply_replacements(code_content, replacement_map):
    for key, val in replacement_map.items():
        if isinstance(val, list):
            continue
        code_content = code_content.replace(key, str(val))
    return code_content


def _replacement_scalar(val, index_var, loop_vars):
    if isinstance(val, list):
        if index_var not in loop_vars:
            raise ValueError(
                f"Loop variable '{index_var}' is not set for per-index replacement"
            )
        idx = loop_vars[index_var]
        if idx < 0 or idx >= len(val):
            raise ValueError(
                f"Index {idx} out of range for replacement list of length {len(val)}"
            )
        return int(val[idx])
    return int(val)


def _lookup_loop_bound(value_str, replacement_map, loop_vars):
    if value_str in replacement_map:
        val = replacement_map[value_str]
        index_match = re.search(r'\[(\w+)\]\s*$', value_str)
        index_var = index_match.group(1) if index_match else None
        return _replacement_scalar(val, index_var, loop_vars)

    indexed_suffix = re.search(r'(\w+\[\w+\])\s*$', value_str)
    if indexed_suffix:
        suffix = indexed_suffix.group(1)
        for key, val in replacement_map.items():
            if key == suffix or key.endswith('.' + suffix):
                index_match = re.search(r'\[(\w+)\]\s*$', key)
                index_var = index_match.group(1) if index_match else None
                return _replacement_scalar(val, index_var, loop_vars)

    return None


def resolve_loop_bound(value_str, replacement_map, loop_vars=None):
    loop_vars = loop_vars or {}
    value_str = ' '.join(value_str.split())
    if value_str.isdigit():
        return int(value_str)
    bound = _lookup_loop_bound(value_str, replacement_map, loop_vars)
    if bound is not None:
        return bound
    raise ValueError(f"Unknown loop bound '{value_str}' in structure file")


def find_matching_brace(code, brace_start):
    depth = 0
    for i in range(brace_start, len(code)):
        if code[i] == '{':
            depth += 1
        elif code[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unmatched '{' while parsing structure file")


def expand_c_for_loops(code, replacement_map, loop_vars=None):
    """Unroll C for-loops using scalar and per-index replacement_map bounds."""
    loop_vars = loop_vars if loop_vars is not None else {}
    out = []
    pos = 0
    while pos < len(code):
        for_match = FOR_LOOP_START_RE.match(code, pos)
        if for_match:
            loop_var = for_match.group(1)
            start_index = int(for_match.group(2))
            op = for_match.group(3)
            end_index = resolve_loop_bound(
                for_match.group(4), replacement_map, loop_vars
            )
            if op == '<':
                end_index -= 1
            elif op == '>':
                end_index += 1

            brace_start = code.find('{', for_match.end())
            brace_end = find_matching_brace(code, brace_start)
            body = code[brace_start + 1:brace_end]
            for i in range(start_index, end_index + 1):
                child_vars = {**loop_vars, loop_var: i}
                out.append(expand_c_for_loops(body, replacement_map, child_vars))
            pos = brace_end + 1
            continue

        next_for = FOR_LOOP_START_RE.search(code, pos)
        if next_for:
            out.append(code[pos:next_for.start()])
            pos = next_for.start()
        else:
            out.append(code[pos:])
            break

    return ''.join(out)


def preprocess_structure_code(code_content, replacement_map):
    code_content = apply_replacements(code_content, replacement_map)
    return expand_c_for_loops(code_content, replacement_map)
