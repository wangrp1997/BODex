import os 
import numpy as np
from copy import deepcopy
import argparse
from curobo.util_file import (
    get_manip_configs_path,
    get_output_path,
    load_yaml,
    write_yaml,
)
import subprocess
import multiprocessing
import datetime

def worker(gpu_id, output_path, cmd):
    with open(output_path, 'w') as output_file:
        subprocess.call(f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}", shell=True, stdout=output_file, stderr=output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--manip_cfg_file",
        type=str,
        default='fc_leap.yml',
        help="config file path",
    )
    
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "-g",
        "--gpu",
        nargs='+',
        required=True,
        help='gpu id list'
    )
    args = parser.parse_args()
    
    manip_config_data = load_yaml(os.path.join(get_manip_configs_path(), args.manip_cfg_file))
    
    if manip_config_data['world']['selection']['id'] is not None:
        id_lst = deepcopy(manip_config_data['world']['selection']['id'])
        original_start = 0
        original_end = len(manip_config_data['world']['selection']['id'])
    else:
        original_start = deepcopy(manip_config_data['world']['selection']['start'])
        original_end = deepcopy(manip_config_data['world']['selection']['end'])
    all_obj_num = original_end - original_start
    obj_num_lst = np.array([all_obj_num // len(args.gpu)] * len(args.gpu))
    obj_num_lst[:(all_obj_num % len(args.gpu))] += 1
    assert obj_num_lst.sum() == all_obj_num
    
    p_list = []
    save_folder = os.path.join(args.manip_cfg_file[:-4], args.path)
    runinfo_folder = os.path.join(get_output_path(), save_folder.replace('graspdata', 'runinfo'))
    os.makedirs(runinfo_folder, exist_ok=True) 
    
    metric_types = ['qp', 'qpbase', 'tdg', 'dfc', 'ch_q1']
    cmd_lst = [f'python example_grasp/eval_energy.py -c {args.manip_cfg_file} -p {args.path} -t {metric}' for metric in metric_types]

    # create tmp manip cfg files
    for i, cmd in enumerate(cmd_lst):
        output_path = os.path.join(runinfo_folder, str(i) + '_eval.txt')
        gpu_id = args.gpu[i % len(args.gpu)]
        p = multiprocessing.Process(target=worker, args=(gpu_id, output_path, cmd))
        p.start()
        print(f"create process :{p.pid}")
        p_list.append(p)
        
    for p in p_list:
        p.join()    