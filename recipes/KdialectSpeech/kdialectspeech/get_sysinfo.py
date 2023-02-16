#!/usr/bin/env python

'''
실행법 : python run_pipe.py run_pipe.yaml

speechbrain/utils/superpowers.py 사용 검토
'''

import torch
from speechbrain.utils.superpowers import run_shell


### cpu info
cpu_cmd = "lscpu"
cpu_out, cpu_err, cpu_code = run_shell(cpu_cmd)

cpu_out_lines = cpu_out.decode().split('\n')
cpu_out_lines_not_nul = list(filter(bool, cpu_out_lines))

cpu_out_dict = dict((k.strip(), v.strip()) 
            for k, v in (line.split(":") 
                for line in cpu_out_lines_not_nul))

print()
print("----- System Info -----")
print("===== CPU Info :")
print(f"CPU Model : {cpu_out_dict['Model name']}")

print(f"CPU MHz : {cpu_out_dict['CPU MHz']}")
print(f"CPU max MHz : {cpu_out_dict['CPU max MHz']}")
print(f"CPU min MHz : {cpu_out_dict['CPU min MHz']}")

print(f"Number of Sockets : {cpu_out_dict['Socket(s)']}")
print(f"Core(s) per socket : {cpu_out_dict['Core(s) per socket']}")
print(f"Thread(s) per core : {cpu_out_dict['Thread(s) per core']}")

num_of_cores = int(cpu_out_dict['Socket(s)'])\
                *int(cpu_out_dict['Core(s) per socket'])\
                *int(cpu_out_dict['Thread(s) per core'])
print(f"Total Number of cores(socket x core x thread) : {num_of_cores}")

print(f"On-line CPU(s) list : {cpu_out_dict['On-line CPU(s) list']}")

### Memory Info
mem_cmd = "free -h | grep Mem"
mem_out, mem_err, mem_code = run_shell(mem_cmd)
mem_out_put = mem_out.decode().split()

print()
print("===== Memory Info :")
print(f"Memory Total : {mem_out_put[1]}")

### gpu info
# print(torch.cuda.is_available())
print()
print("===== GPU Info :")
if torch.cuda.is_available():
    gpu_num = torch.cuda.device_count()
    for gpu in range(gpu_num):
        gpu_model = torch.cuda.get_device_name(gpu)
        mem = torch.cuda.get_device_properties(gpu).total_memory/(1024 * 1024 * 1024)
        print(f'GPU {gpu} - model: {gpu_model}, gpu memory: {mem:.2f} GB')
else:
    print(f'There is no GPU')



