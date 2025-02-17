import argparse
import multiprocessing
import subprocess
import os 
import numpy as np

def worker(gpu_id, folder, start, end, output_path):
    with open(output_path, 'w') as output_file:
        subprocess.call(f"python example_grasp/assets_process/render_pc.py -d {folder} -g {gpu_id} -s {start} -e {end}", shell=True, stdout=output_file, stderr=output_file)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_dir', 
        type=str, 
        default='/mnt/disk1/jiayichen/data/dex_obj/meshdata', 
        help='the path to the data directory'
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=10,
        help='number on each gpu'
    )
    parser.add_argument(
        "-g",
        "--gpu",
        nargs='+',
        required=True,
        help='gpu id list'
    )
    args = parser.parse_args()
    
    gpu_lst = args.gpu * args.number
    
    all_obj_num = len(os.listdir(args.data_dir))
    obj_num_lst = np.array([all_obj_num // len(gpu_lst)] * len(gpu_lst))
    obj_num_lst[:(all_obj_num % len(gpu_lst))] += 1
    assert obj_num_lst.sum() == all_obj_num
    
    p_list = []
    for i, gpu_id in enumerate(gpu_lst):
        start = (obj_num_lst[:i]).sum()
        end = (obj_num_lst[:(i+1)]).sum()
        
        os.makedirs('debug', exist_ok=True)
        output_path = f'debug/render{i}.txt'
        p = multiprocessing.Process(target=worker, args=(gpu_id, args.data_dir, start, end, output_path))
        p.start()
        print(f"create process :{p.pid}")
        p_list.append(p)
        
    for p in p_list:
        p.join()    
    
    
    