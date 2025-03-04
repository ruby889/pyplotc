#!/home/ruby/miniconda3/envs/mujoco/bin/python
import pandas as pd
import matplotlib.pyplot as plt
from pygnuplot import gnuplot
import re
from collections import defaultdict
import sys

plot_tool = 'gnuplot'
# plot_tool = 'matplotlib'
replace_map = {"Robot.JointSize": "7", "rs.JointSize": "7", "ms.JointSize": "7", "Driver.DriverSize[arm_i]": "7", "camera_size": "2", "gripper_size":"1"}
def readStruct(structure_file):
    structure = defaultdict(list)
    structure_index = {}
    structure_name = {}
    index = []
    cur_i = 0
    #Analysis c++/c fprintf function
    with open(structure_file, "r") as f:
        for_loop_param = (0,1)
        prev_line = ''
        for line0 in f:
            try:
                #Incomplete line
                line = prev_line + line0.strip()
                if line and line[-1] != ';' and line[-1] != '{' and line[-1] != ')':
                    prev_line += line
                    continue
                
                for key, val in replace_map.items():
                    line = line.replace(key, val)
                    
                split_line = re.sub(r"[\{\}]", '', line)        #remove { & }
                split_line = re.split('\(|\)', split_line)      #split by ( & )
                if split_line[0].strip() == "fprintf":
                    context = split_line[1].split(',')
                    if len(context) < 3:    #If no parameters in fprintf
                        continue
                    
                    cnt = context[1].count('%')
                    names = [re.sub('^.*(\.|->)', '', x) for x in context[2:]] #remove string before . or ->
                    names = [re.sub('\[.*', '', x) for x in names]  #remove all [*]. e.g. torque[i] -> torque
                    #Save names
                    for i in range(cnt):
                        structure_name[i+cur_i] = names[i]
                        structure_index[names[i]] = i+cur_i

                    #Save data index
                    for i in range(*for_loop_param):
                        for j in range(cur_i, cnt+cur_i):
                            structure[j].append(len(index))
                            index.append(j)
                    cur_i += cnt*(for_loop_param[1]-for_loop_param[0])
                    for_loop_param = (0,1)
                elif split_line[0].strip() == "for":
                    params = split_line[1].split(';')
                    for_loop_param = (int(params[0].strip()[-1]), int(params[1].strip()[-1]))
                prev_line = ''
            except:
                print("line0: ", line0) 
                print("line: ", line)
                print("split_line: ", split_line)
                raise Exception("Error orccurs")
    return structure, structure_index, structure_name

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
            cmd += f"'{filename}'using {x+1} w l title '{x+1}: {names[i].strip()}' noenhanced, "
        g.cmd(cmd)
    input("Press Enter to continue...")

if __name__ == '__main__':
    '''
    Analysis sys.argv
    e.g. python3 plot_data.py [structure_file] [data_file] [{options}] [plot data]               : plot two graphs, 0&3 in same graph, 5 in another graph
    e.g. python3 plot_data.py ./writeFileStructure.txt ~/33.txt 0,3 5                            : plot two graphs, 0&3 in same graph, 5 in another graph
    e.g. python3 plot_data.py ./writeFileStructure.txt ~/33.txt joint_pos                        : plot joint_pos
    e.g. python3 plot_data.py ./writeFileStructure.txt ~/33.txt                                  : plot all graphs
    e.g. python3 plot_data.py ./writeFileStructure.txt ~/33.txt {"set xrange": [200:]} joint_pos : plot joint_pos with xrange set to [200:] 
    '''
    if len(sys.argv) <= 2: 
        print("Wrong Args")
    else:
        #Handle data struct
        structure_file = sys.argv[1]
        structure, structure_index, structure_name = readStruct(structure_file)

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
                    pattern = re.compile(r'\[(\d+):(\d*)\]')
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
                    key = int(x) if x.isnumeric() else structure_index[x]
                    temp.append(key)
                graphs.append(temp)

        for graph in graphs:
            graph_indexs.append([])
            graph_names.append("")
            graph_titles.append("")
            for line in graph:
                graph_indexs[-1].extend(structure[line])
                name = structure_name[line] + ", "
                graph_names[-1] += name * len(structure[line])
                graph_titles[-1] += name
            graph_titles[-1] = graph_titles[-1][:-2]

        if (plot_tool == 'gnuplot'):
            plotWithGnuplot(graph_indexs, graph_names, graph_titles, graph_options, data_file)
        elif (plot_tool == 'matplotlib'):
            plotWithMatplotlib(graph_indexs, graph_names, graph_titles, graph_options, df)
    
    