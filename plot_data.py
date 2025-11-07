#!/home/ruby/miniconda3/envs/mujoco/bin/python
import argparse
from collections import defaultdict
import json
import re
import os
import sys
from enum import Enum

from pycparser import c_parser, c_ast
import pandas as pd
import matplotlib.pyplot as plt
from pygnuplot import gnuplot

class CCodeVisitor(c_ast.NodeVisitor):
    """AST visitor to extract fprintf statements and for loops from C code."""
    
    def __init__(self, replacement_map):
        self.replacement_map = replacement_map
        self.fprintf_calls = []
        
    def visit_FuncCall(self, node):
        """Visit function calls to find fprintf statements."""
        if hasattr(node.name, 'name') and node.name.name == 'fprintf':
            self.extract_fprintf_info(node)
        self.generic_visit(node)
        
    def visit_For(self, node):
        """Visit for loops to extract loop parameters."""
        for_info = self.extract_for_info(node)
        parent_fprintf_calls = self.fprintf_calls.copy()
        self.fprintf_calls = []
        self.generic_visit(node)
        self.fprintf_calls = parent_fprintf_calls + [{"for_context": self.fprintf_calls, "for_info":for_info}]
        
    def extract_fprintf_info(self, node):
        """Extract information from fprintf function calls."""
        if not node.args or len(node.args.exprs) < 2:
            return
            
        # Get format string to count parameters
        format_arg = node.args.exprs[1]
        if hasattr(format_arg, 'value'):
            format_str = format_arg.value.strip('"')
            param_count = format_str.count('%')
        else:
            param_count = 0
            
        # Extract parameter names
        param_names = []
        if len(node.args.exprs) > 2:
            for arg in node.args.exprs[2:]:
                name = self.extract_variable_name(arg)
                if name:
                    param_names.append(name)
                    
        fprintf_info = {
            'param_count': param_count,
            'param_names': param_names,
        }
        self.fprintf_calls.append(fprintf_info)
        
    def extract_for_info(self, node):
        """Extract for loop information."""
        start_index = 0
        end_index = 1
        
        # Extract initialization
        if node.init:
            if hasattr(node.init, 'rvalue') and hasattr(node.init.rvalue, 'value'):
                start_index = int(node.init.rvalue.value)
                
        # Extract condition
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
        """Extract variable name from AST node."""
        if isinstance(node, c_ast.ID):
            return node.name
        elif isinstance(node, c_ast.ArrayRef):
            # For array references like set_velocity[i] or ptr->field[i], get the base name
            if isinstance(node.name, c_ast.ID):
                return node.name.name
            elif isinstance(node.name, c_ast.ArrayRef):
                # Handle cases like ptr->field[i][j]
                return self.extract_variable_name(node.name)
            elif isinstance(node.name, c_ast.StructRef):
                # Handle cases like ptr->field[i] - extract the field name
                return self.extract_variable_name(node.name)
            elif hasattr(node.name, 'name'):
                return node.name.name
            else:
                return str(node.name)
        elif isinstance(node, c_ast.StructRef):
            # For struct references like ptr->field, get the field name
            if isinstance(node.field, c_ast.ID):
                return node.field.name
            elif hasattr(node.field, 'name'):
                return node.field.name
            else:
                return str(node.field)
        elif hasattr(node, 'name'):
            if isinstance(node.name, c_ast.ID):
                return node.name.name
            elif hasattr(node.name, 'name'):
                return node.name.name
            else:
                return str(node.name)
        elif hasattr(node, 'expr') and hasattr(node.expr, 'name'):
            return node.expr.name
        else:
            return str(node)

def parse_c_code_with_pycparser(code_content, replacement_map):
    """Parse C code using pycparser and extract fprintf and for loop information."""
    try:
        # Preprocess the code to handle macros
        for key, val in replacement_map.items():
            code_content = code_content.replace(key, str(val))
            
        # Create a minimal C function wrapper without includes
        # pycparser needs function declarations for fprintf
        preprocessed_code = """
            int fprintf(void* stream, const char* format, ...);
            void dummy_function() {
            """ + code_content + """
            }
        """
        
        # Parse the code
        parser = c_parser.CParser()
        ast = parser.parse(preprocessed_code)
        
        # Visit the AST to extract information
        visitor = CCodeVisitor(replacement_map)
        visitor.visit(ast)
        
        return visitor.fprintf_calls
        
    except Exception as e:
        print(f"Error parsing C code with pycparser: {e}")
        # Fallback to regex-based parsing
        return [], []

def parse_array_target(target_str, structure_by_name, structure_by_index, replacement_map, column_mode=False):
    """Parse target string that may contain array indexing.
    
    Examples:
    - 'set_velocity[2]' -> returns column index for set_velocity[2]
    - 'set_velocity[1][3]' -> returns column index for set_velocity[1][3]
    - 'set_velocity' -> returns all column indices for set_velocity
    - '2' -> returns column index 2 (or variable in column 2 if not column_mode)
    
    Args:
        column_mode: If True, numbers refer to direct column indices. If False, numbers refer to variable names.
    
    Returns list of column indices.
    """
    target_str = target_str.strip()
    
    # Handle numeric column index
    if target_str.isdigit():
        col_index = int(target_str)
        if column_mode:
            # Direct column indexing
            if col_index < len(structure_by_index):
                return [col_index]
            else:
                print(f"Warning: Column index {col_index} out of range (max: {len(structure_by_index)-1})")
                return []
        else:
            # Legacy behavior: look for variable name in that column
            if col_index < len(structure_by_index):
                var_name = structure_by_index[col_index]
                if var_name in structure_by_name:
                    return structure_by_name[var_name]
                else:
                    return [col_index]  # Fallback to direct column
            else:
                print(f"Warning: Column index {col_index} out of range (max: {len(structure_by_index)-1})")
                return []
    
    # Handle array indexing like set_velocity[2] or set_velocity[1][3]
    if '[' in target_str and ']' in target_str:
        # Extract variable name and indices
        match = re.match(r'(\w+)((?:\[\d+\])+)', target_str)
        if match:
            var_name = match.group(1)
            indices_str = match.group(2)
            
            # Extract all indices from brackets
            indices = [int(x) for x in re.findall(r'\[(\d+)\]', indices_str)]
            
            # Find the matching column index
            if var_name in structure_by_name:
                # We need to find which column corresponds to this specific array element
                # This requires understanding the array layout from the C code
                return find_array_element_column(var_name, indices, structure_by_name, structure_by_index, replacement_map)
            else:
                print(f"Warning: Variable '{var_name}' not found in structure")
                return []
    
    # Handle simple variable name
    if target_str in structure_by_name:
        return structure_by_name[target_str]
    else:
        print(f"Warning: Variable '{target_str}' not found in structure")
        return []

def find_array_element_column(var_name, target_indices, structure_by_name, structure_by_index, replacement_map=None):
    """Find the column index for a specific array element.
    
    Uses the replacement map to determine actual array dimensions from the C code.
    """
    if var_name not in structure_by_name:
        return []
    
    all_indices = structure_by_name[var_name]
    total_elements = len(all_indices)
    
    if len(target_indices) == 1:
        # 1D array: target_indices[0] is the element index
        element_index = target_indices[0]
        if element_index < total_elements:
            return [all_indices[element_index]]
        else:
            print(f"Warning: Array index {element_index} out of range for {var_name} (max: {total_elements-1})")
            return []
    elif len(target_indices) == 2:
        # 2D array: target_indices[0] is row, target_indices[1] is column
        # Try to determine dimensions from replacement map
        if replacement_map:
            # Look for common 2D array patterns in the replacement map
            if var_name == 'set_velocity' and 'NUM_ARM' in replacement_map and 'JOINTS_PER_ARM' in replacement_map:
                rows = int(replacement_map['NUM_ARM'])
                cols = int(replacement_map['JOINTS_PER_ARM'])
            elif 'NUM_JOINTS' in replacement_map:
                # For other arrays, assume they're 1D with NUM_JOINTS elements
                rows = 1
                cols = int(replacement_map['NUM_JOINTS'])
            else:
                # Fallback: assume square-ish array
                rows = int(total_elements ** 0.5) + 1
                cols = total_elements // rows + (1 if total_elements % rows else 0)
        else:
            # Fallback: assume square-ish array
            rows = int(total_elements ** 0.5) + 1
            cols = total_elements // rows + (1 if total_elements % rows else 0)
        
        row, col = target_indices
        if row < rows and col < cols:
            element_index = row * cols + col
            if element_index < total_elements:
                return [all_indices[element_index]]
            else:
                print(f"Warning: Calculated index {element_index} out of range for {var_name} (max: {total_elements-1})")
                return []
        else:
            print(f"Warning: Array indices [{row}][{col}] out of range for {var_name} (max: [{rows-1}][{cols-1}])")
            return []
    
    print(f"Warning: Unsupported array indexing for {var_name}{target_indices}")
    return []

def read_replacement_map(replacement_file):
    with open(replacement_file, 'r') as file:
        replacement_map = json.load(file)
        
    for key, val in replacement_map.items():
        if not isinstance(val, str):
            replacement_map[key] = str(val)
    return replacement_map

def extract_fprintf_calls(fprintf_calls):
    ordered_params = []
    
    for fprintf_info in fprintf_calls:
        if "for_context" in fprintf_info:
            nested_ordered_params = extract_fprintf_calls(fprintf_info["for_context"])
            start, end = fprintf_info["for_info"]['start'], fprintf_info["for_info"]['end']
            loop_ordered_params = []
            for _ in range(start, end+1):
                loop_ordered_params += nested_ordered_params
            ordered_params.extend(loop_ordered_params)
        else:
            param_count = fprintf_info['param_count']
            param_names = fprintf_info['param_names']
            
            # Clean parameter names
            cleaned_names = []
            for name in param_names:
                # Convert to string if it's an AST node
                if not isinstance(name, str):
                    name = str(name)
                
                # Clean up AST node string representations
                if name.startswith("ID(name='") and name.endswith("')"):
                    name = name[9:-2]  # Remove "ID(name='" and "')"
                elif "ID(name=" in name:
                    # Handle multiline AST representations
                    match = re.search(r"ID\(name='([^']+)'", name)
                    if match:
                        name = match.group(1)
                
                # Remove prefixes like obj. or obj->
                name = re.sub(r'^.*(\.|->)', '', name)
                # Remove array indices like [i]
                name = re.sub(r'\[.*\]', '', name)
                # Remove trailing characters and whitespace
                name = name.strip('();{}\n ')
                cleaned_names.append(name)
            
            for j, name in enumerate(cleaned_names[:param_count]):
                ordered_params.append(name)
    return ordered_params
        

def read_struct(structure_file, replacement_map):
    structure_by_name = {}
    
    # Read the entire C code file
    with open(structure_file, "r") as f:
        code_content = f.read()
    
    # Try to parse with pycparser first
    fprintf_calls = parse_c_code_with_pycparser(code_content, replacement_map)
    ordered_param_indexes = extract_fprintf_calls(fprintf_calls)
    for i, name in enumerate(ordered_param_indexes):
        if name not in structure_by_name:
            structure_by_name[name] = []
        structure_by_name[name].append(i)
    return structure_by_name, ordered_param_indexes

def plot_with_matplotlib(figure_indexs, figure_names, figure_titles, frame_indexs, df):
    start, end = frame_indexs
    for figure_i in range(len(figure_indexs)):
        labels = list(zip(figure_names[figure_i], figure_indexs[figure_i]))
        labels = [f"{index}: {name}" for name, index in labels]
        df.iloc[start:end, figure_indexs[figure_i]].plot(title=figure_titles[figure_i])
        plt.legend(labels)
    plt.show()

def plot_with_gnuplot(figure_indexs, figure_names, figure_titles, frame_indexs, data_file_path):
    start, end = frame_indexs
    figures = []
    for figure_i in range(len(figure_indexs)):
        g = gnuplot.Gnuplot()
        figures.append(g)
        g.cmd(f'set xrange [{start}:{end}]')
        g.cmd(f'set title "{figure_titles[figure_i]}" noenhanced')

        cmd = 'plot '
        names = figure_names[figure_i]
        for i, x in enumerate(figure_indexs[figure_i]):
            filename = data_file_path if i == 0 else ""
            cmd += f"'{filename}' using {x+1} w l title '{x}: {names[i]}' noenhanced, "
        g.cmd(cmd)
    input("Press Enter to continue...")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot data from structure and data files with optional settings."
    )

    parser.add_argument("--structure_file",     type=str, help="Path to the structure file")
    parser.add_argument("--data_file",          type=str, help="Path to the data file")
    parser.add_argument("--replacement_file",   type=str, default='replacement.json', help="Path to the JSON file for macro/variable replacement in the structure file")
    parser.add_argument("--target",             type=str, help="Comma-separated list of columns or variable names to plot. Use ':' to separate different graphs. Supports array indexing: 'set_velocity[2]' or 'set_velocity[1][3]' for specific elements.")
    parser.add_argument("--column_mode",        action="store_true", help="When enabled, numbers in target refer to direct column indices instead of variable names")
    parser.add_argument("--start_frame",        type=int, default=0, help="Start index of frame")
    parser.add_argument("--end_frame",          type=int, default=0, help="Last index of frame")
    parser.add_argument("--plot_tool",          type=str, default='gnuplot', help="Choose 'gnuplot' or 'matplotlib' for plotting (default: gnuplot)")

    args = parser.parse_args()

    data_file_path = os.path.expanduser(args.data_file)
    if not data_file_path:
        raise ValueError("data_file path cannot be None.")
    elif not os.path.exists(data_file_path):
        raise FileNotFoundError(f"data_file '{data_file_path}' does not exist.")

    structure_file_path = os.path.expanduser(args.structure_file)
    if not structure_file_path:
        raise ValueError("structure_file path cannot be None.")
    elif not os.path.exists(structure_file_path):
        raise FileNotFoundError(f"structure_file '{structure_file_path}' does not exist.")
     
    return {
        "structure_file": structure_file_path,
        "data_file": data_file_path,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "target": args.target,
        "column_mode": args.column_mode,
        "plot_tool": args.plot_tool,
        "replacement_file": args.replacement_file
    }

if __name__ == '__main__':
    args = parse_arguments()
    
    #Add replacement map
    replacement_map = read_replacement_map(args["replacement_file"])

    structure_file_path = args["structure_file"]
    data_file_path = args["data_file"]
    print(f"Structure file: {structure_file_path}")
    print(f"Data file: {data_file_path}")
    if args["column_mode"]:
        print("Column mode enabled: Numbers in target refer to direct column indices")
    
    #Handle data struct
    structure_by_name, structure_params_array = read_struct(args["structure_file"], replacement_map)
    print(f"\nstructure_params_array: {structure_params_array}\n")
    print(f"structure_by_name: {structure_by_name}\n")
    
    #Handle data
    df = pd.read_csv(data_file_path, sep=" ", low_memory=False)

    #Handle options
    start_frame = args["start_frame"]
    end_frame = args["end_frame"] if args["end_frame"] != 0 else df.shape[0]
    frame_indexs = (start_frame, end_frame)

    #Handle plot info
    figure_indexs, figure_names, figure_titles, group_names_per_figure = [], [], [], []
    if not args["target"]: #Plot all data
        group_names_per_figure = [[graph] for graph in structure_by_name.keys()]
    else: #Plot data specified by user
        targets_per_figure = args["target"].split(":")
        for targets in targets_per_figure:
            group_indices = []
            group_names = []
            for target in targets.split(','): 
                target = target.strip()
                # Parse target (may include array indexing)
                indices = parse_array_target(target, structure_by_name, structure_params_array, replacement_map, args["column_mode"])
                if indices:
                    group_indices.extend(indices)
                    # Get variable name for display
                    if target.isdigit():
                        var_name = structure_params_array[int(target)]
                    elif '[' in target:
                        # Extract variable name from array notation
                        var_name = target.split('[')[0]
                    else:
                        var_name = target
                    group_names.extend([var_name] * len(indices))
            group_names_per_figure.append(group_indices)

    for group_indices in group_names_per_figure:
        indexs = []
        names = []
        titles = []
        if isinstance(group_indices[0], int):
            # group_indices contains actual column indices
            indexs = group_indices
            names = [structure_params_array[i] for i in group_indices]
            titles = list(set(names))  # Unique variable names for title
        else:
            # Legacy format - group_names contains variable names
            for group_name in group_indices:
                all_indexs = structure_by_name[group_name]
                n = len(all_indexs)
                indexs.extend(all_indexs)
                names.extend([group_name]*n)
                titles.append(group_name)
        figure_indexs.append(indexs)
        figure_names.append(names)
        figure_titles.append(', '.join(titles))

    print("group_names_per_figure: ", group_names_per_figure)
    print("figure_indexs: ", figure_indexs)
    print("figure_names: ", figure_names)
    print("figure_titles: ", figure_titles)
    #plot data
    if (args["plot_tool"] == 'gnuplot'):
        plot_with_gnuplot(figure_indexs, figure_names, figure_titles, frame_indexs, data_file_path)
    elif (args["plot_tool"] == 'matplotlib'):
        plot_with_matplotlib(figure_indexs, figure_names, figure_titles, frame_indexs, df)
    
    