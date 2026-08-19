import argparse
import json
import re
import os

import pandas as pd
import matplotlib.pyplot as plt
from pygnuplot import gnuplot

from fprintf_structure import parse_fprintf_structure
from oss_structure import is_oss_structure, parse_oss_structure

def filter_by_index(all_column_indices, index_filter, target_label=""):
    if index_filter is None:
        return all_column_indices

    filtered = []
    for idx in index_filter:
        if idx < len(all_column_indices):
            filtered.append(all_column_indices[idx])
        else:
            label = f" for '{target_label}'" if target_label else ""
            print(
                f"Warning: index {idx} out of range{label} "
                f"(max: {len(all_column_indices) - 1})"
            )
    return filtered


def parse_index(index_str):
    if not index_str:
        return None
    return [int(value.strip()) for value in index_str.split(',') if value.strip()]


def parse_array_target(target_str, structure_by_name, structure_by_index, replacement_map, index_filter=None):
    """Parse target string that may contain array indexing.

    When index_filter is set (e.g. [0, 1]), each expanded same-name variable list
    is limited to those flat positions. Not global column numbers. Explicit bracket
    syntax such as set_velocity[2] is unchanged.
    """
    target_str = target_str.strip()

    if target_str.isdigit():
        col_index = int(target_str)
        if col_index < len(structure_by_index):
            var_name = structure_by_index[col_index]
            if var_name in structure_by_name:
                return filter_by_index(
                    structure_by_name[var_name], index_filter, target_str
                )
            return [col_index]
        print(f"Warning: Column index {col_index} out of range (max: {len(structure_by_index)-1})")
        return []

    if '[' in target_str and ']' in target_str:
        match = re.match(r'(\w+)((?:\[\d+\])+)', target_str)
        if match:
            var_name = match.group(1)
            indices = [int(x) for x in re.findall(r'\[(\d+)\]', match.group(2))]

            if var_name in structure_by_name:
                return find_array_element_column(
                    var_name, indices, structure_by_name, structure_by_index
                )
            print(f"Warning: Variable '{var_name}' not found in structure")
            return []

    if target_str in structure_by_name:
        return filter_by_index(
            structure_by_name[target_str], index_filter, target_str
        )

    print(f"Warning: Variable '{target_str}' not found in structure")
    return []

def find_array_element_column(var_name, target_indices, structure_by_name, structure_by_index):
    """Find the column index for one array element by flat 0-based index."""
    if var_name not in structure_by_name:
        return []

    all_indices = structure_by_name[var_name]
    total_elements = len(all_indices)

    if len(target_indices) != 1:
        print(
            f"Warning: use flat 0-based indexing for '{var_name}', "
            f"e.g. {var_name}[2] (not {var_name}{''.join(f'[{i}]' for i in target_indices)})"
        )
        return []

    element_index = target_indices[0]
    if element_index < total_elements:
        return [all_indices[element_index]]

    print(f"Warning: Array index {element_index} out of range for {var_name} (max: {total_elements-1})")
    return []

def read_replacement_map(replacement_file):
    with open(replacement_file, 'r') as file:
        replacement_map = json.load(file)
        
    for key, val in replacement_map.items():
        if not isinstance(val, str):
            replacement_map[key] = str(val)
    return replacement_map

def build_structure_map(ordered_param_indexes):
    structure_by_name = {}
    for i, name in enumerate(ordered_param_indexes):
        if name not in structure_by_name:
            structure_by_name[name] = []
        structure_by_name[name].append(i)
    return structure_by_name, ordered_param_indexes

def read_columns_from_data_header(data_file_path):
    with open(data_file_path, 'r', encoding='utf-8', errors='replace') as file:
        first_line = file.readline()
    if not first_line.startswith('#'):
        return None
    columns = first_line.lstrip('#').strip().split()
    return columns if columns else None

def read_data_file(data_file_path, skip_header=False):
    kwargs = dict(sep=r'\s+', header=None, low_memory=False)
    if skip_header:
        kwargs['skiprows'] = 1
    return pd.read_csv(data_file_path, **kwargs)

def resolve_column_structure(structure_file, data_file_path, replacement_map):
    header_columns = read_columns_from_data_header(data_file_path)
    has_data_header = header_columns is not None

    if structure_file:
        structure_by_name, structure_params_array = read_struct(structure_file, replacement_map)
        source = f"structure file: {structure_file}"
    elif header_columns:
        structure_by_name, structure_params_array = build_structure_map(header_columns)
        source = f"data file header ({len(header_columns)} columns)"
    else:
        raise ValueError(
            "No column mapping available. Provide --structure_file or use a data file "
            "whose first line is a '#' column header."
        )

    return structure_by_name, structure_params_array, has_data_header, source

def read_struct(structure_file, replacement_map):
    with open(structure_file, "r") as f:
        code_content = f.read()

    if is_oss_structure(code_content):
        ordered_param_indexes = parse_oss_structure(code_content, replacement_map)
    else:
        ordered_param_indexes = parse_fprintf_structure(code_content, replacement_map)

    return build_structure_map(ordered_param_indexes)

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

def resolve_data_file_path(data_file, data_directory=None):
    data_file_path = os.path.expanduser(data_file)
    if not data_file_path:
        raise ValueError("data_file path cannot be None.")
    if data_directory:
        data_file_path = os.path.join(os.path.expanduser(data_directory), data_file_path)
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"data_file '{data_file_path}' does not exist.")
    return data_file_path

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot data from structure and data files with optional settings."
    )

    parser.add_argument("--structure_file",     type=str, default=None, help="Path to the structure file")
    parser.add_argument("--data_file",          type=str, help="Path to the data file (relative to --data_directory when set)")
    parser.add_argument("--data_directory",     type=str, default=None, help="Optional directory prepended to --data_file")
    parser.add_argument("--replacement_file",   type=str, default='replacement.json', help="Path to the JSON file for macro/variable replacement in the structure file")
    parser.add_argument("--target",             type=str, help="Comma-separated list of columns or variable names to plot. Use ':' to separate different graphs. Supports flat array indexing: 'set_velocity[2]' picks the 0-based flattened element 2.")
    parser.add_argument("--index",              type=str, default=None, help="Optional comma-separated flat indices (e.g. '0,1') applied per same-name variable to limit expanded columns. Not global column numbers. Explicit targets like set_velocity[2] are unchanged.")
    parser.add_argument("--start_frame",        type=int, default=0, help="Start index of frame")
    parser.add_argument("--end_frame",          type=int, default=0, help="Last index of frame")
    parser.add_argument("--plot_tool",          type=str, default='gnuplot', help="Choose 'gnuplot' or 'matplotlib' for plotting (default: gnuplot)")

    args = parser.parse_args()

    data_file_path = resolve_data_file_path(args.data_file, args.data_directory)

    structure_file_path = None
    if args.structure_file:
        structure_file_path = os.path.expanduser(args.structure_file)
        if not os.path.exists(structure_file_path):
            raise FileNotFoundError(f"structure_file '{structure_file_path}' does not exist.")
     
    return {
        "structure_file": structure_file_path,
        "data_file": data_file_path,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "target": args.target,
        "index": args.index,
        "plot_tool": args.plot_tool,
        "replacement_file": args.replacement_file
    }

if __name__ == '__main__':
    args = parse_arguments()
    
    #Add replacement map
    replacement_map = read_replacement_map(args["replacement_file"])

    structure_file_path = args["structure_file"]
    data_file_path = args["data_file"]
    print(f"Structure file: {structure_file_path or '(not provided)'}")
    print(f"Data file: {data_file_path}")
    index_filter = parse_index(args["index"])
    if index_filter is not None:
        print(f"Index filter: {index_filter}")

    structure_by_name, structure_params_array, skip_data_header, column_source = resolve_column_structure(
        structure_file_path, data_file_path, replacement_map
    )
    print(f"Using columns from {column_source}")

    print(f"\nstructure_params_array: {structure_params_array}\n")
    print(f"structure_by_name: {structure_by_name}\n")

    df = read_data_file(data_file_path, skip_header=skip_data_header)
    data_col_count = df.shape[1]

    if data_col_count != len(structure_params_array):
        if structure_file_path:
            raise ValueError(
                f"Column count mismatch: data file has {data_col_count} columns, "
                f"but structure file defines {len(structure_params_array)} columns."
            )
        print(
            f"Warning: data file has {data_col_count} columns, header lists "
            f"{len(structure_params_array)} names (max layout). "
            "Only indices within data range are plottable."
        )

    def filter_plottable_indices(indices, target_label=""):
        valid = [i for i in indices if i < data_col_count]
        dropped = [i for i in indices if i >= data_col_count]
        if dropped:
            dropped_names = [structure_params_array[i] for i in dropped]
            label = f" for '{target_label}'" if target_label else ""
            print(
                f"Warning: skipping {len(dropped)} column(s){label} not present in data "
                f"(data has {data_col_count} columns): {dropped_names}"
            )
        return valid

    #Handle options
    start_frame = args["start_frame"]
    end_frame = args["end_frame"] if args["end_frame"] != 0 else df.shape[0]
    frame_indexs = (start_frame, end_frame)

    #Handle plot info
    figure_indexs, figure_names, figure_titles, group_names_per_figure = [], [], [], []
    if not args["target"]: #Plot all data
        group_names_per_figure = [
            filter_plottable_indices(structure_by_name[graph], graph)
            for graph in structure_by_name.keys()
        ]
        group_names_per_figure = [g for g in group_names_per_figure if g]
    else: #Plot data specified by user
        targets_per_figure = args["target"].split(":")
        for targets in targets_per_figure:
            group_indices = []
            group_names = []
            for target in targets.split(','): 
                target = target.strip()
                indices = filter_plottable_indices(
                    parse_array_target(
                        target, structure_by_name, structure_params_array,
                        replacement_map, index_filter
                    ),
                    target,
                )
                if indices:
                    group_indices.extend(indices)
                    if target.isdigit():
                        var_name = structure_params_array[int(target)]
                    elif '[' in target:
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
    
    