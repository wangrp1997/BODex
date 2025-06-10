# Standard Library
import time
import os
import datetime

# Third Party
import torch
import numpy as np
import argparse

# CuRobo
from curobo.geom.sdf.world import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.robot import JointState, RobotConfig
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolverConfig, IKSolver
from curobo.util.world_cfg_generator import get_world_config_dataloader
from curobo.util.save_helper import SaveHelper
from curobo.util.logger import setup_logger, log_warn
from curobo.util.traj_util import interp_move_traj
from curobo.util_file import (
    get_manip_configs_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import random

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


class PlanningFailureError(Exception):
    pass


def get_armhand_qpos(arm_ik: IKSolver, hand_pose_qpos, retract_qpos):
    ik_result = arm_ik.solve_single(
        Pose.from_list(hand_pose_qpos.view(-1)[:7]),
        seed_config=retract_qpos.view(1, 1, -1)[..., : arm_ik.kinematics.dof],
        retract_config=retract_qpos.view(1, -1)[:, : arm_ik.kinematics.dof],
    )
    if not ik_result.success:
        raise PlanningFailureError("IK fails")
    armhand_qpos = torch.cat(
        [ik_result.solution.squeeze(1), hand_pose_qpos.view(1, -1)[:, 7:]], dim=-1
    )
    return armhand_qpos  # [1, n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--manip_cfg_file",
        type=str,
        default="fc_leap.yml",
        help="config file path",
    )

    parser.add_argument(
        "-f",
        "--save_folder",
        type=str,
        default=None,
        help="If None, use join_path(manip_cfg_file[:-4], $TIME) as save_folder",
    )

    parser.add_argument(
        "-m",
        "--save_mode",
        choices=["usd", "npy", "usd+npy", "none"],
        default="npy",
        help="Method to save results",
    )

    parser.add_argument(
        "-k",
        "--skip",
        action="store_false",
        help="If True, skip existing files. (default: True)",
    )

    args = parser.parse_args()

    setup_logger("warn")
    tensor_args = TensorDeviceType()
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))

    if args.save_folder is not None:
        save_folder = os.path.join(args.save_folder, "graspdata")
    elif manip_config_data["exp_name"] is not None:
        save_folder = os.path.join(
            args.manip_cfg_file[:-4], manip_config_data["exp_name"], "graspdata"
        )
    else:
        save_folder = os.path.join(
            args.manip_cfg_file[:-4],
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "graspdata",
        )

    assert manip_config_data["world"]["type"] == "dexonomy"
    log_warn(f"Load dexonomy grasp pose from {manip_config_data['world']['template_path']}!")

    world_generator = get_world_config_dataloader(manip_config_data["world"], 1)

    save_mogen = SaveHelper(
        robot_file=manip_config_data["robot_file_with_arm"],
        save_folder=save_folder,
        task_name="mogen",
        mode=args.save_mode,
    )

    # Initialize hand forward kinematic model
    robot_config_data = load_yaml(
        join_path(get_robot_configs_path(), manip_config_data["robot_file"])
    )["robot_cfg"]
    hand_robocfg = RobotConfig.from_dict(robot_config_data)
    hand_model = CudaRobotModel(hand_robocfg.kinematics)

    tst = time.time()
    mg = None
    arm_iksolver = None
    for world_info_dict in world_generator:
        sst = time.time()
        if args.skip and save_mogen.exist_piece(world_info_dict["save_prefix"]):
            log_warn(f"skip {world_info_dict['save_prefix']}")
            continue

        for k, v in world_info_dict.items():
            if "qpos" in k or k == "move_cfg":
                world_info_dict[k] = v[0]

        world_model = [WorldConfig.from_dict(world_info_dict["world_cfg"][0])]

        if arm_iksolver is None:
            robot_config_data = load_yaml(
                join_path(get_robot_configs_path(), manip_config_data["robot_file_with_arm"])
            )["robot_cfg"]
            ik_link_names = hand_model.transfered_link_name
            robot_config_data["kinematics"]["link_names"] = ik_link_names
            robot_config_data["kinematics"]["ee_link"] = ik_link_names[0]
            ik_robot_cfg = RobotConfig.from_dict(robot_config_data, tensor_args)
            ik_config = IKSolverConfig.load_from_robot_config(
                ik_robot_cfg,
                world_model[0],
                tensor_args=tensor_args,
                high_precision=True,
            )
            arm_iksolver = IKSolver(ik_config)
        else:
            arm_iksolver.update_world(world_model)

        pregrasp_qpos = tensor_args.to_device(world_info_dict["pregrasp_qpos"])
        grasp_qpos = tensor_args.to_device(world_info_dict["grasp_qpos"])
        squeeze_qpos = tensor_args.to_device(world_info_dict["squeeze_qpos"])

        pregrasp_pose, pregrasp_joint = pregrasp_qpos[:, :7], pregrasp_qpos[:, 7:]
        grasp_pose, grasp_joint = grasp_qpos[:, :7], grasp_qpos[:, 7:]
        squeeze_pose, squeeze_joint = squeeze_qpos[:, :7], squeeze_qpos[:, 7:]

        if mg is None:
            motion_gen_cfg = MotionGenConfig.load_from_robot_config(
                manip_config_data["robot_file_with_arm"],
                world_model=world_model[0],
                collision_activation_distance=0.025,
            )
            mg = MotionGen(motion_gen_cfg)
        else:
            mg.update_world(world_model[0])

        try:
            # Plan approaching trajectory
            init_arm_qpos = tensor_args.to_device(manip_config_data["mogen_init"]).view(1, -1)
            pregrasp_ah_qpos_lst = [init_arm_qpos.clone()]
            for i in reversed(range(len(pregrasp_qpos))):
                pregrasp_ah_qpos_lst.append(
                    get_armhand_qpos(arm_iksolver, pregrasp_qpos[i], pregrasp_ah_qpos_lst[-1])
                )
            pregrasp_qpos = torch.flip(torch.cat(pregrasp_ah_qpos_lst[1:], dim=0), dims=[0])
            pregrasp_vel = (pregrasp_qpos[0:1] - pregrasp_qpos[1:2]) / mg.interpolation_dt / 10
            
            target_state = JointState.from_numpy(
                position=pregrasp_qpos[0:1],
                velocity=pregrasp_vel,
                joint_names=mg.kinematics.joint_names,
            )
            init_state = JointState.from_position(init_arm_qpos)

            mogen_result = mg.plan_single_js(
                start_state=target_state,
                goal_state=init_state,
                plan_config=MotionGenPlanConfig(
                    enable_finetune_trajopt=False, num_trajopt_seeds=4, max_attempts=1
                ),
            )
            if sum(mogen_result.success) == 0:
                raise PlanningFailureError(f"MoGen fails")
            approach_qpos = torch.flip(mogen_result.optimized_plan.position[2:], [0])

            # grasp and squeeze qpos
            grasp_qpos = get_armhand_qpos(arm_iksolver, grasp_qpos, pregrasp_qpos[-1])
            squeeze_qpos = get_armhand_qpos(arm_iksolver, squeeze_qpos, grasp_qpos)

            # move qpos
            move_pose_lst = mg.tensor_args.to_device(
                interp_move_traj(squeeze_pose[0].cpu().numpy(), world_info_dict["move_cfg"], 10)
            )
            move_qpos_lst = [squeeze_qpos.clone()]
            for i in range(len(move_pose_lst)):
                tmp_qpos = torch.cat(
                    [move_pose_lst[i], squeeze_qpos[0, arm_iksolver.kinematics.dof :]], dim=-1
                )
                move_qpos_lst.append(get_armhand_qpos(arm_iksolver, tmp_qpos, move_qpos_lst[-1]))
            move_qpos = torch.cat(move_qpos_lst[1:], dim=0)

            # Check smoothness
            stage2_qpos = torch.cat([pregrasp_qpos, grasp_qpos, squeeze_qpos, move_qpos], dim=-2)
            if (stage2_qpos[1:, :8] - stage2_qpos[:-1, :8]).abs().max() > 0.1:
                raise PlanningFailureError("Non-smooth IK fails")
        except PlanningFailureError as e:
            log_warn(e, world_info_dict["save_prefix"][0])
            continue
        if "usd" in args.save_mode:
            world_info_dict["robot_pose"] = torch.cat([approach_qpos, stage2_qpos], dim=-2)

        world_info_dict["approach_qpos"] = approach_qpos
        world_info_dict["pregrasp_qpos"] = pregrasp_qpos
        world_info_dict["grasp_qpos"] = grasp_qpos
        world_info_dict["squeeze_qpos"] = squeeze_qpos
        world_info_dict["move_qpos"] = move_qpos

        world_info_dict["world_model"] = world_model
        save_mogen.save_piece(world_info_dict)
        log_warn(f"Sinlge Time (mogen): {time.time()-sst}")
    log_warn(f"Total Time: {time.time()-tst}")
