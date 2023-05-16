#!/home/ruby/miniconda3/envs/mujoco/bin/python
import pandas as pd
import matplotlib.pyplot as plt
from pygnuplot import gnuplot
import re
from collections import defaultdict
import sys

structure_file = "writeFileStructure.txt"
data_file = "~/robotics/build/33.txt"

def useMatplotlib(graph_indexs, graph_names):
    for graph_i in range(len(graph_indexs)):
        print(graph_indexs[graph_i], graph_names[graph_i])
        df.iloc[:, graph_indexs[graph_i]].plot(title=graph_names[graph_i])
        plt.legend(graph_indexs[graph_i])
    plt.show()

def useGnuplot(graph_indexs, graph_names):
    for graph_i in range(len(graph_indexs)):
        g = gnuplot.Gnuplot()
        g.cmd(f'set title "{graph_names[graph_i]}"')

        cmd = 'plot '
        for i, x in enumerate(graph_indexs[graph_i]):
            filename = data_file if i == 0 else ""
            cmd += f'"{filename}"using {x+1} w l, '
        g.cmd(cmd)
    input("Press Enter to continue...")

if __name__ == '__main__':
    structure = defaultdict(list)
    structure_index = {}
    structure_name = {}
    index = []
    cur_i = 0
    #Analysis c++/c fprintf function
    with open(structure_file, "r") as f:
        for_loop_param = (0,1)
        prev_line = ''
        for line in f:
            #Incomplete line
            line = prev_line + line.strip()
            if line[-1] != ';' and line[-1] != '{' and line[-1] != ')':
                prev_line += line
                continue
            
            split_line = re.sub(r"[\{\}]", '', line) #remove { & }
            split_line = re.split('\(|\)', split_line)    #split by ( & )
            if split_line[0].strip() == "fprintf":
                context = split_line[1].split(',')
                if len(context) < 3:    #If only string is fprintf
                    continue
                cnt = context[1].count('%')
                names = [re.sub('^.*(\.|->)', '', x) for x in context[2:]] #remove all characters before . or ->
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

    '''
    Analysis sys.argv
    e.g. plot_data.py 0,3 5  : plot two graphs, 0&3 in same graph, 5 in another graph
    e.g. plot_data.py Target : plot graph by its name
    '''
    df = pd.read_csv(data_file, sep=" ")
    graph_indexs, graph_names = [], []
    if len(sys.argv) == 1:
        keys = structure.keys()
        graph_indexs = structure[keys]
        graph_names = structure_name[keys]
    else:
        for cmd in sys.argv[1:]:
            graph_indexs.append([])
            graph_names.append("")
            for key in cmd.split(','): 
                key = int(key) if key.isnumeric() else structure_index[key]
                graph_indexs[-1].extend(structure[key])
                graph_names[-1] += structure_name[key] + ", "
            graph_names[-1] = graph_names[-1][:-2] #Remove last ','
    
    # Plot with gnuplot
    useGnuplot(graph_indexs, graph_names)

    # # Plot with matplotlib
    # useMatplotlib(graph_indexs, graph_names)
    
    