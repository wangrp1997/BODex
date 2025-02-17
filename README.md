# BODex

A GPU-based efficient pipeline for dexterous grasp synthesis built on [cuRobo](https://github.com/NVlabs/curobo/tree/main), proposed in *BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization [ICRA 2025]*.


## Getting Started
1. Installation.
```
conda create -n bodex python=3.10
conda activate bodex

conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia 

pip install -e . --no-build-isolation  

pip uninstall numpy
pip install numpy==1.26.4
pip install usd-core 
conda install pytorch-scatter -c pyg
conda install pinocchio -c conda-forge

cd src/curobo/geom/cpp
python setup.py install    # install hppfcl_openmp_wrapper
```

2. Object preparing and processing. See [MeshProcess](https://github.com/JYChen18/MeshProcess) for details. Create soft links for the processed data folders.
```
ln -s ${YOUR_DATA_PATH} src/curobo/content/assets/object/DGNObj  
ln -s ${YOUR_SPLIT_PATH} src/curobo/content/assets/object/DGNObj_splits
```
   
3. Synthesize grasp poses.
```
# Single GPU version
CUDA_VISIBLE_DEVICES=7 python example_grasp/plan_batch_env.py -c sim_shadow/fc.yml -w 40 

# Multiple GPU version
python example_grasp/multi_gpu.py -c sim_shadow/fc.yml -m grasp -g 0 1 2 3 
```
We can also synthesize approaching trajectories with the motion planning of cuRobo.
 ```
# Single GPU version
CUDA_VISIBLE_DEVICES=7 python example_grasp/plan_mogen_batch.py -c sim_shadow/tabletop.yml -t grasp_and_mogen

# Multiple GPU version
python example_grasp/multi_gpu.py -c sim_shadow/tabletop.yml -m grasp_and_mogen -g 0 1 2 3 
```
The grasp synthesis supports to parallize different objects, but the motion planning only supports to parallize different trajectories for the same object.

4. (Optional) Visualization.
```
python example_grasp/visualize_npy.py -c sim_shadow/fc.yml -f debug
```

## Project using BODex
Many projects make modifications based on our pipeline to synthesize large-scale datasets of grasping poses, such as 
- [DexGraspNet2.0](https://pku-epic.github.io/DexGraspNet2.0/)
<!-- - [GraspVLA (coming soon)]() -->

## Citation

If you found this repository useful, please consider to cite the following works,

```
@misc{curobo_report23,
      title={cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation},
      author={Balakumar Sundaralingam and Siva Kumar Sastry Hari and Adam Fishman and Caelan Garrett
              and Karl Van Wyk and Valts Blukis and Alexander Millane and Helen Oleynikova and Ankur Handa
              and Fabio Ramos and Nathan Ratliff and Dieter Fox},
      year={2023},
      eprint={2310.17274},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
@article{chen2024bodex,
  title={BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization},
  author={Chen, Jiayi and Ke, Yubin and Wang, He},
  journal={arXiv preprint arXiv:2412.16490},
  year={2024}
}
```
