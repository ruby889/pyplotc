#!/home/ruby/miniconda3/envs/mujoco/bin/python
import pandas as pd
import matplotlib.pyplot as plt
from pygnuplot import gnuplot
import re
from collections import defaultdict
import sys

structure_file = "writeFileStructure.txt"
data_file = "~/Desktop/33.txt"
plot_tool = 'gnuplot'
replace_map = {"Robot.JointSize": "7"}
def useMatplotlib(graph_indexs, graph_names):
    for graph_i in range(len(graph_indexs)):
        print(graph_indexs[graph_i], graph_names[graph_i])
        df.iloc[:, graph_indexs[graph_i]].plot(title=graph_names[graph_i])
        plt.legend(graph_indexs[graph_i])
    plt.show()

def useGnuplot(graph_indexs, graph_names):
    for graph_i in range(len(graph_indexs)):
        name = graph_names[graph_i].replace('_', '\\\_')
        g = gnuplot.Gnuplot()
        g.cmd(f'set title "{name}"')

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
    e.g. python3 plot_data.py 0,3 5     : plot two graphs, 0&3 in same graph, 5 in another graph
    e.g. python3 plot_data.py joint_pos : plot joint_pos in graph
    e.g. python3 plot_data.py           : plot all graphs
    '''
    df = pd.read_csv(data_file, sep=" ")
    graph_indexs, graph_names = [], []
    if len(sys.argv) == 1:
        keys = structure.keys()
        graph_indexs = list(structure.values())
        graph_names = list(structure_name.values())
    else:
        for cmd in sys.argv[1:]:
            graph_indexs.append([])
            graph_names.append("")
            for x in cmd.split(','): 
                key = int(x) if x.isnumeric() else structure_index[x]
                graph_indexs[-1].extend(structure[key])
                graph_names[-1] += structure_name[key] + ", "
            graph_names[-1] = graph_names[-1][:-2] #Remove last ','

    if (plot_tool == 'gnuplot'):
        # Plot with gnuplot
        useGnuplot(graph_indexs, graph_names)
    elif (plot_tool == 'matplotlib'):
        # # Plot with matplotlib
        useMatplotlib(graph_indexs, graph_names)
    
    