#!/home/ruby/miniconda3/envs/mujoco/bin/python
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import sys
if __name__ == '__main__':
    structure_file = "writeFileStructure.txt"
    data_file = "~/robotics/build/33.txt"

    structure = defaultdict(list)
    structure_name = {}
    index = []
    cur_i = 0
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

    df = pd.read_csv(data_file, sep=" ")
    if (sys.argv == 'all' or len(sys.argv) == 1):
        keys = structure.keys()
    else:
        keys = [int(x) for x in sys.argv[1].split(',')]

    for key in keys:
        df.iloc[:, structure[key]].plot(title=structure_name[key])
        plt.legend(structure[key])
    plt.show()
    
    