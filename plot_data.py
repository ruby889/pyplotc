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
        self.for_loops = []
        self.for_loop_stack = []  # Stack to track nested loops
        
    def visit_FuncCall(self, node):
        """Visit function calls to find fprintf statements."""
        if hasattr(node.name, 'name') and node.name.name == 'fprintf':
            self.extract_fprintf_info(node)
        self.generic_visit(node)
        
    def visit_For(self, node):
        """Visit for loops to extract loop parameters."""
        for_info = self.extract_for_info(node)
        self.for_loop_stack.append(for_info)
        self.for_loops.append(for_info)
        self.generic_visit(node)
        self.for_loop_stack.pop()
        
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
            'for_context': list(self.for_loop_stack)  # Copy current nested loop context
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
            # For array references like set_velocity[i], get the base name
            if isinstance(node.name, c_ast.ID):
                return node.name.name
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
        
        return visitor.fprintf_calls, visitor.for_loops
        
    except Exception as e:
        print(f"Error parsing C code with pycparser: {e}")
        # Fallback to regex-based parsing
        return [], []

def read_replacement_map(replacement_file):
    with open(replacement_file, 'r') as file:
        replacement_map = json.load(file)
        
    for key, val in replacement_map.items():
        if not isinstance(val, str):
            replacement_map[key] = str(val)
    return replacement_map

def read_struct(structure_file, replacement_file):
    replacement_map = read_replacement_map(replacement_file)
    structure_by_name = {}
    structure_by_index = {}
    cur_i = 0
    
    # Read the entire C code file
    with open(structure_file, "r") as f:
        code_content = f.read()
    
    # Try to parse with pycparser first
    fprintf_calls, for_loops = parse_c_code_with_pycparser(code_content, replacement_map)
    
    if fprintf_calls:
        # Use pycparser results
        for fprintf_info in fprintf_calls:
            param_count = fprintf_info['param_count']
            param_names = fprintf_info['param_names']
            for_context = fprintf_info['for_context']
            
            # Calculate total iterations for nested loops
            total_iterations = 1
            if for_context:
                for loop_info in for_context:
                    loop_size = loop_info['end'] - loop_info['start'] + 1
                    total_iterations *= loop_size
                
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
                    import re
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
            
            # Initialize structure for names
            for name in cleaned_names[:param_count]:
                if name not in structure_by_name:
                    structure_by_name[name] = []
            
            # Add data indices for nested loops
            for i in range(total_iterations):
                for j, name in enumerate(cleaned_names[:param_count]):
                    structure_by_index[cur_i] = name
                    structure_by_name[name].append(cur_i)
                    cur_i += 1
    return structure_by_name, structure_by_index

def plot_with_matplotlib(figure_indexs, figure_names, figure_titles, frame_indexs, df):
    start, end = frame_indexs
    for figure_i in range(len(figure_indexs)):
        labels = list(zip(figure_names[figure_i], figure_indexs[figure_i]))
        labels = [f"{name}_{index}" for name, index in labels]
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
            cmd += f"'{filename}' using {x+1} w l title '{x+1}: {names[i]}' noenhanced, "
        g.cmd(cmd)
    input("Press Enter to continue...")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot data from structure and data files with optional settings."
    )

    parser.add_argument("--structure_file",     type=str, help="Path to the structure file")
    parser.add_argument("--data_file",          type=str, help="Path to the data file")
    parser.add_argument("--replacement_file",   type=str, default='replacement.json', help="Path to the JSON file for macro/variable replacement in the structure file")
    parser.add_argument("--target",             type=str, help="Comma-separated list of columns or variable names to plot. Use ':' to separate different graphs.")
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
        "plot_tool": args.plot_tool,
        "replacement_file": args.replacement_file
    }

if __name__ == '__main__':
    args = parse_arguments()
    
    #Handle data struct
    structure_file_path = args["structure_file"]
    structure_by_name, structure_by_index = read_struct(args["structure_file"], args["replacement_file"])
    print(f"Structure file: {structure_file_path}")
    
    #Handle data
    data_file_path = args["data_file"]
    df = pd.read_csv(data_file_path, sep=" ", low_memory=False)
    print(f"Data file: {data_file_path}")

    #Handle options
    start_frame = args["start_frame"]
    end_frame = args["end_frame"] if args["end_frame"] != 0 else df.shape[0]
    frame_indexs = (start_frame, end_frame)

    #Handle plot info
    figure_indexs, figure_names, figure_titles, group_names_per_figure = [], [], [], []
    if not args["target"]: #Plot all data
        group_names_per_figure = [[graph] for graph in structure.keys()]
    else: #Plot data specified by user
        targets_per_figure = args["target"].split(":")
        for targets in targets_per_figure:
            group_names = []
            for line in targets.split(','): 
                group_name = structure_by_index[int(line)] if line.isnumeric() else line
                group_names.append(group_name)
            group_names_per_figure.append(group_names)

    for group_names in group_names_per_figure:
        indexs = []
        names = []
        titles = []
        for group_name in group_names:
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
    
    