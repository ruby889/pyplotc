"""Parse std::ostringstream telemetry data-writing structure definitions."""

import re

from structure_preprocess import preprocess_structure_code

STREAM_STMT_RE = re.compile(r'\w+\s*<<\s*(.*?);', re.DOTALL)
FPRINTF_RE = re.compile(r'\bfprintf\s*\(')
OSTRINGSTREAM_RE = re.compile(r'\bostringstream\b')
STREAM_DATA_RE = re.compile(r'<<\s*\w+\.')
STREAM_ARRAY_RE = re.compile(r'<<\s*\w+\s*\[')
STREAM_CHAR_SEP_RE = re.compile(r"<<\s*'\s'")
STREAM_HEADER_RE = re.compile(r'<<\s*"#\s+\w')
CHAR_LITERAL_RE = re.compile(r"^'(?:\\.|[^'\\])'$")
STRING_LITERAL_RE = re.compile(r'^"(?:\\.|[^"\\])*"$')


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


def extract_oss_stream_params(code):
    """Collect column names from oss << statements in already-unrolled code."""
    ordered_params = []
    pos = 0
    while pos < len(code):
        while pos < len(code) and code[pos].isspace():
            pos += 1
        if pos >= len(code):
            break

        stream_match = STREAM_STMT_RE.match(code, pos)
        if stream_match:
            ordered_params.extend(extract_stream_data_names(stream_match.group(1)))
            pos = stream_match.end()
            continue

        next_stream = STREAM_STMT_RE.search(code, pos)
        if not next_stream:
            break
        pos = next_stream.start()

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
    code_content = preprocess_structure_code(code_content, replacement_map)
    return extract_oss_stream_params(code_content)
