"""Parse std::ostringstream telemetry data-writing structure definitions."""

import re

FOR_LOOP_START_RE = re.compile(
    r'for\s*\(\s*int\s+(\w+)\s*=\s*(\d+)\s*;\s*\1\s*(<=|<|>=|>)\s*([^;]+?)\s*;'
)
STREAM_STMT_RE = re.compile(r'\w+\s*<<\s*(.*?);', re.DOTALL)
FPRINTF_RE = re.compile(r'\bfprintf\s*\(')
OSTRINGSTREAM_RE = re.compile(r'\bostringstream\b')
STREAM_DATA_RE = re.compile(r'<<\s*\w+\.')
STREAM_ARRAY_RE = re.compile(r'<<\s*\w+\s*\[')
STREAM_CHAR_SEP_RE = re.compile(r"<<\s*'\s'")
STREAM_HEADER_RE = re.compile(r'<<\s*"#\s+\w')
CHAR_LITERAL_RE = re.compile(r"^'(?:\\.|[^'\\])'$")
STRING_LITERAL_RE = re.compile(r'^"(?:\\.|[^"\\])*"$')


def apply_replacements(code_content, replacement_map):
    for key, val in replacement_map.items():
        code_content = code_content.replace(key, str(val))
    return code_content


def resolve_loop_bound(value_str, replacement_map):
    value_str = ' '.join(value_str.split())
    if value_str.isdigit():
        return int(value_str)
    if value_str in replacement_map:
        return int(replacement_map[value_str])
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
    raise ValueError("Unmatched '{' while parsing oss structure")


def clean_variable_name(expr):
    expr = expr.strip()
    expr = re.sub(r'^.*(\.|->)', '', expr)
    expr = re.sub(r'\[.*\]', '', expr)
    return expr.strip()


def extract_stream_data_names(stream_body):
    """Extract column names from one oss << data statement."""
    parts = re.split(r'\s*<<\s*', stream_body.strip())
    names = []

    for part in parts:
        part = part.strip().rstrip(',').strip()
        if not part:
            continue
        if CHAR_LITERAL_RE.match(part) or STRING_LITERAL_RE.match(part):
            if part.strip("'\"") in (' ', '\n', ''):
                continue
            continue
        name = clean_variable_name(part)
        if name:
            names.append(name)

    return names


def expand_oss_code(code, replacement_map):
    ordered_params = []
    pos = 0
    while pos < len(code):
        while pos < len(code) and code[pos].isspace():
            pos += 1
        if pos >= len(code):
            break

        for_match = FOR_LOOP_START_RE.match(code, pos)
        if for_match:
            start_index = int(for_match.group(2))
            op = for_match.group(3)
            end_index = resolve_loop_bound(for_match.group(4), replacement_map)
            if op == '<':
                end_index -= 1
            elif op == '>':
                end_index += 1

            brace_start = code.find('{', for_match.end())
            brace_end = find_matching_brace(code, brace_start)
            body = code[brace_start + 1:brace_end]
            for _ in range(start_index, end_index + 1):
                ordered_params.extend(expand_oss_code(body, replacement_map))
            pos = brace_end + 1
            continue

        stream_match = STREAM_STMT_RE.match(code, pos)
        if stream_match:
            ordered_params.extend(extract_stream_data_names(stream_match.group(1)))
            pos = stream_match.end()
            continue

        next_for = FOR_LOOP_START_RE.search(code, pos)
        next_stream = STREAM_STMT_RE.search(code, pos)
        candidates = [match.start() for match in (next_for, next_stream) if match]
        if not candidates:
            break
        pos = min(candidates)

    return ordered_params


def is_oss_structure(code_content):
    if FPRINTF_RE.search(code_content):
        return False
    if STREAM_HEADER_RE.search(code_content):
        return False
    if STREAM_DATA_RE.search(code_content):
        return True
    if STREAM_ARRAY_RE.search(code_content):
        return True
    if STREAM_CHAR_SEP_RE.search(code_content):
        return True
    if OSTRINGSTREAM_RE.search(code_content) and re.search(r"<<\s*'\s'", code_content):
        return True
    return False


def parse_oss_structure(code_content, replacement_map):
    code_content = apply_replacements(code_content, replacement_map)
    return expand_oss_code(code_content, replacement_map)
