"""Parse fprintf telemetry structure definitions."""

import re

from pycparser import c_parser, c_ast


class CCodeVisitor(c_ast.NodeVisitor):
    """AST visitor to extract fprintf statements and for loops from C code."""

    def __init__(self, replacement_map):
        self.replacement_map = replacement_map
        self.fprintf_calls = []

    def visit_FuncCall(self, node):
        if hasattr(node.name, 'name') and node.name.name == 'fprintf':
            self.extract_fprintf_info(node)
        self.generic_visit(node)

    def visit_For(self, node):
        for_info = self.extract_for_info(node)
        parent_fprintf_calls = self.fprintf_calls.copy()
        self.fprintf_calls = []
        self.generic_visit(node)
        self.fprintf_calls = parent_fprintf_calls + [{"for_context": self.fprintf_calls, "for_info": for_info}]

    def extract_fprintf_info(self, node):
        if not node.args or len(node.args.exprs) < 2:
            return

        format_arg = node.args.exprs[1]
        if hasattr(format_arg, 'value'):
            format_str = format_arg.value.strip('"')
            param_count = format_str.count('%')
        else:
            param_count = 0

        param_names = []
        if len(node.args.exprs) > 2:
            for arg in node.args.exprs[2:]:
                name = self.extract_variable_name(arg)
                if name:
                    param_names.append(name)

        self.fprintf_calls.append({
            'param_count': param_count,
            'param_names': param_names,
        })

    def extract_for_info(self, node):
        start_index = 0
        end_index = 1

        if node.init:
            if hasattr(node.init, 'rvalue') and hasattr(node.init.rvalue, 'value'):
                start_index = int(node.init.rvalue.value)

        if node.cond:
            if hasattr(node.cond, 'right') and hasattr(node.cond.right, 'value'):
                end_value = int(node.cond.right.value)
                op = node.cond.op
                if op == '<':
                    end_index = end_value - 1
                elif op == '<=':
                    end_index = end_value
                elif op == '>':
                    end_index = end_value + 1
                elif op == '>=':
                    end_index = end_value
                else:
                    end_index = end_value

        return {'start': start_index, 'end': end_index}

    def extract_variable_name(self, node):
        if isinstance(node, c_ast.ID):
            return node.name
        if isinstance(node, c_ast.ArrayRef):
            if isinstance(node.name, c_ast.ID):
                return node.name.name
            if isinstance(node.name, c_ast.ArrayRef):
                return self.extract_variable_name(node.name)
            if isinstance(node.name, c_ast.StructRef):
                return self.extract_variable_name(node.name)
            if hasattr(node.name, 'name'):
                return node.name.name
            return str(node.name)
        if isinstance(node, c_ast.StructRef):
            if isinstance(node.field, c_ast.ID):
                return node.field.name
            if hasattr(node.field, 'name'):
                return node.field.name
            return str(node.field)
        if hasattr(node, 'name'):
            if isinstance(node.name, c_ast.ID):
                return node.name.name
            if hasattr(node.name, 'name'):
                return node.name.name
            return str(node.name)
        if hasattr(node, 'expr') and hasattr(node.expr, 'name'):
            return node.expr.name
        return str(node)


def parse_c_code_with_pycparser(code_content, replacement_map):
    try:
        for key, val in replacement_map.items():
            code_content = code_content.replace(key, str(val))

        preprocessed_code = """
            int fprintf(void* stream, const char* format, ...);
            void dummy_function() {
            """ + code_content + """
            }
        """

        parser = c_parser.CParser()
        ast = parser.parse(preprocessed_code)
        visitor = CCodeVisitor(replacement_map)
        visitor.visit(ast)
        return visitor.fprintf_calls

    except Exception as e:
        print(f"Error parsing C code with pycparser: {e}")
        return []


def extract_fprintf_calls(fprintf_calls):
    ordered_params = []

    for fprintf_info in fprintf_calls:
        if "for_context" in fprintf_info:
            nested_ordered_params = extract_fprintf_calls(fprintf_info["for_context"])
            start, end = fprintf_info["for_info"]['start'], fprintf_info["for_info"]['end']
            loop_ordered_params = []
            for _ in range(start, end + 1):
                loop_ordered_params += nested_ordered_params
            ordered_params.extend(loop_ordered_params)
        else:
            param_count = fprintf_info['param_count']
            param_names = fprintf_info['param_names']

            cleaned_names = []
            for name in param_names:
                if not isinstance(name, str):
                    name = str(name)

                if name.startswith("ID(name='") and name.endswith("')"):
                    name = name[9:-2]
                elif "ID(name=" in name:
                    match = re.search(r"ID\(name='([^']+)'", name)
                    if match:
                        name = match.group(1)

                name = re.sub(r'^.*(\.|->)', '', name)
                name = re.sub(r'\[.*\]', '', name)
                name = name.strip('();{}\n ')
                cleaned_names.append(name)

            for name in cleaned_names[:param_count]:
                ordered_params.append(name)

    return ordered_params


def parse_fprintf_structure(code_content, replacement_map):
    fprintf_calls = parse_c_code_with_pycparser(code_content, replacement_map)
    return extract_fprintf_calls(fprintf_calls)
