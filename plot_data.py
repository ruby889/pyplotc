#!/home/ruby/miniconda3/envs/mujoco/bin/python
import pandas as pd
import matplotlib.pyplot as plt
from pygnuplot import gnuplot
import re
from collections import defaultdict
import sys
from enum import Enum

plot_tool = 'gnuplot'
# plot_tool = 'matplotlib'
replacement_map = {"arm_size_"                  :"2", 
                   "joint_size_"                :"7", 
                   "DriverSize"                 :"7", 
                   "ArmSize"                    :"2", 
                   "Robot.JointSize"            :"7", 
                   "rs.JointSize"               :"7", 
                   "ms.JointSize"               :"7", 
                   "Driver.DriverSize[arm_i]"   :"7", 
                   "camera_size"                :"0",
                   "gripper_size"               :"2"}

def process_for_loop(lines, row_index, col_index):
    structure = defaultdict(list)
    sub_cnt, sub_structure = 0, {}
    cnt = 0

    cmd = lines[row_index]
    pattern = r"for\s*\(\s*(?:[\w<>]+)?\s*(\w+)\s*=\s*([^;]+);\s*(.*?)\s*;\s*(.*?)\s*\)"
    match = re.search(pattern, lines[row_index])
    if match:    
        loop_var = match.group(1)  # Captures loop variable
        start_index = match.group(2).strip()  # Captures initialization value
        end_condition = match.group(3).strip()  # Captures loop condition
        increment = match.group(4).strip()  # Captures increment operation

        pattern1 = r"\s*\w+\s*(<|<=)\s*(\w+\.*\w*)"
        match1 = re.search(pattern1, end_condition)
        operator = match1.group(1) 
        end_index_str = match1.group(2)
        end_index_str = re.sub(r'^.*(\.|->)', '', end_index_str) #remove string before . or ->
        for key, val in replacement_map.items():
            end_index_str = end_index_str.replace(key, val)

        start_index = int(start_index)
        end_index = int(end_index_str)
        end_index += 0 if (operator == '<') else 1
        increment_val = -1 if ('--' in increment) else 1
        # print("Start Index:", start_index)
        # print("End Index:", end_index)
        # print("increment_val", increment_val)
    else:
        raise Exception(f"Cannot breakdown for statement: {lines[row_index]}")
    row_index += 1

    while row_index < len(lines):
        line = lines[row_index].strip()
        if line[:3] == "for":
            row_index, temp_cnt, temp_structure = process_for_loop(lines, row_index, sub_cnt)
            sub_structure.update(temp_structure)
            sub_cnt += temp_cnt
            continue
        elif line[:7] == "fprintf":
            row_index, temp_cnt, temp_structure = process_fprintf(lines, row_index, sub_cnt)
            sub_structure.update(temp_structure)
            sub_cnt += temp_cnt
            continue
        else:
            #Continue read line if it not yet finished
            if not line or line[-1] != '}':
                cmd += lines[row_index]
                row_index += 1
                continue
            for i in range(start_index, end_index, increment_val):
                for key, arr in sub_structure.items():
                    structure[key].extend([col_index + i*sub_cnt + x for x in arr])

            cnt = (end_index-start_index)*sub_cnt
            braket_i = line.find('}')
            if (braket_i != len(line) - 1):
                lines[row_index] = line[braket_i:]
                return row_index, cnt, structure
            return row_index + 1, cnt, structure


def process_fprintf(lines, row_index, col_index):
    cmd = ""
    structure = defaultdict(list)
    while row_index < len(lines):
        line = lines[row_index].strip()
        cmd += line
        #Continue read line if it not yet finished
        if (not ';' in line):
            row_index += 1
            continue
        else:
            split_lines = re.split(r',(?![^<]*>)', cmd)
            format_cnt = 2
            if len(split_lines) > 2:                #If we have parameters in fprintf
                format_str = split_lines[1]
                format_cnt = format_str.count('%')
                names = split_lines[2:]
                names = [re.sub(r'^.*(\.|->)', '', x) for x in split_lines[2:]] #remove string before . or ->
                names = [re.sub(r'\[.*', '', x) for x in names]  #remove all [*]. e.g. torque[i] -> torque
                names = [x.strip(");}") for x in names] #Remove trailing '}' and ';' and ')'
                names = [x.strip() for x in names] #Remove leading and trailing spaces
                if format_cnt != len(names):
                    raise Exception("format_cnt != len(names) from cmd: ", cmd)
                for i, name in enumerate(names):
                    structure[name].append(col_index + i)

            semicolon_i = line.find(';')
            if (semicolon_i != len(line) - 1):
                lines[row_index] = line[semicolon_i:]
                return row_index, format_cnt, structure
            return row_index+1, format_cnt, structure
    raise Exception("process_fprintf met invalid cmd: ", cmd)

def readStruct(structure_file):
    structure = defaultdict(list)
    structure_names = []
    current_col_index = 0

    #Analysis c++/c fprintf function
    with open(structure_file, "r") as f:
        lines = f.readlines()
    row_index = 0
    try:
        while row_index < len(lines):
            line = lines[row_index].strip() #Remove spaces
            if not line: 
                row_index += 1
                continue
            if line[:7] == "fprintf":
                row_index, cnt, temp_structure = process_fprintf(lines, row_index, current_col_index)
            elif line[:3] == "for":
                row_index, cnt, temp_structure = process_for_loop(lines, row_index, current_col_index)
            else:
                raise Exception("Unknown starting string") 
            for key, val in temp_structure.items():
                structure[key].extend(val)
            current_col_index += cnt
    except Exception as e:
        print(f"row_index: {row_index}, line: {line}")
        raise e
    
    structure_names = [0]*current_col_index
    for key, val in structure.items():
        for x in val:
            structure_names[x] = key
    return structure, structure_names

def plotWithMatplotlib(graph_indexs, graph_names, graph_titles, graph_options, df):
    start, end = 0, df.shape[0]
    if "xrange" in graph_options:
        start,end = graph_options["xrange"]

    for graph_i in range(len(graph_indexs)):
        df.iloc[start:end, graph_indexs[graph_i]].plot(title=graph_titles[graph_i])
        plt.legend(graph_indexs[graph_i])
    plt.show()

def plotWithGnuplot(graph_indexs, graph_names, graph_titles, graph_options, filename):
    start, end = 0, df.shape[0]
    if "xrange" in graph_options:
        start,end = graph_options["xrange"]

    for graph_i in range(len(graph_indexs)):
        g = gnuplot.Gnuplot()
        if "xrange" in graph_options:
            start,end = graph_options["xrange"]
            g.cmd(f'set xrange [{start}:{end}]')
        g.cmd(f'set title "{graph_titles[graph_i]}" noenhanced')

        cmd = 'plot '
        names = graph_names[graph_i].split(',')
        for i, x in enumerate(graph_indexs[graph_i]):
            filename = data_file if i == 0 else ""
            cmd += f"'{filename}' using {x+1} w l title '{x+1}: {names[i].strip()}' noenhanced, "
        g.cmd(cmd)
    input("Press Enter to continue...")

if __name__ == '__main__':
    '''
    Analysis sys.argv
    e.g. python3 plot_data.py [structure_file] [data_file] [{options}] [plot data]               : plot two graphs, 0&3 in same graph, 5 in another graph
    e.g. python3 plot_data.py ./writeFileStructure.txt ~/33.txt 0,3 5                            : plot two graphs, 0&3 in same graph, 5 in another graph
    e.g. python3 plot_data.py ./writeFileStructure.txt ~/33.txt joint_pos                        : plot joint_pos
    e.g. python3 plot_data.py ./writeFileStructure.txt ~/33.txt                                  : plot all graphs
    e.g. python3 plot_data.py ./writeFileStructure.txt ~/33.txt {"xrange":[200:]} joint_pos : plot joint_pos with xrange set to [200:] 
    '''
    if len(sys.argv) <= 2: 
        print("Wrong Args")
    else:
        #Handle data struct
        structure_file = sys.argv[1]
        structure, structure_names = readStruct(structure_file)

        #Handle data
        data_file = sys.argv[2]
        df = pd.read_csv(data_file, sep=" ", low_memory=False)

        #Handle options
        graph_options = {}
        next_pos = 3
        if len(sys.argv) > 3 and sys.argv[3][0] == '{':
            next_pos += 1
            options = sys.argv[3][1:-1].split(',')
            for opt in options:
                key, val = opt.split(':', 1)
                if key == "xrange":
                    pattern = re.compile(r'\[(\d*):(\d*)\]')
                    match = pattern.match(val)
                    start = int(match.group(1)) if match.group(1) else 0
                    end = int(match.group(2)) if match.group(2) else df.shape[0]
                    graph_options[key] = (start, end)

        #Handle plot info
        graph_indexs, graph_names, graph_titles, graphs = [], [], [], []
        if len(sys.argv) == next_pos: #Plot all data
            graphs = [[graph] for graph in structure.keys()]
        else: #Plot data specified by user
            for cmd in sys.argv[next_pos:]:
                temp = []
                for x in cmd.split(','): 
                    key = x if not x.isnumeric() else structure_names[int(x)]
                    temp.append(key)
                graphs.append(temp)
                
        for graph in graphs:
            graph_indexs.append([])
            graph_names.append("")
            graph_titles.append("")
            for name in graph:
                graph_indexs[-1].extend(structure[name])
                if (len(structure[name]) > 1):
                    graph_names[-1] += "".join([f"{name}_{i}, " for i in range(len(structure[name]))])
                else:
                    graph_names[-1] += f"{name}, "
                graph_titles[-1] += name if not graph_titles[-1] else ', ' + name
            graph_titles[-1] = graph_titles[-1]

        if (plot_tool == 'gnuplot'):
            plotWithGnuplot(graph_indexs, graph_names, graph_titles, graph_options, data_file)
        elif (plot_tool == 'matplotlib'):
            plotWithMatplotlib(graph_indexs, graph_names, graph_titles, graph_options, df)
    
    