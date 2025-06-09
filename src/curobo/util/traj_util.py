import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def np_normalize_vector(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def np_interp_slide(pose1: np.ndarray, pose2: np.ndarray, step: int) -> np.ndarray:
    trans1, quat1 = pose1[:3], pose1[3:7]
    trans2, quat2 = pose2[:3], pose2[3:7]
    slerp = Slerp([0, 1], R.from_quat([quat1, quat2], scalar_first=True))
    trans_interp = np.linspace(trans1, trans2, step + 1)[1:]
    quat_interp = slerp(np.linspace(0, 1, step + 1))[1:].as_quat(scalar_first=True)
    return np.concatenate([trans_interp, quat_interp], axis=1)


def np_interp_hinge(pose1, hinge_pos, hinge_axis, move_angle, step):
    """
    pose1: (7,) initial pose (translation and quaternion)
    hinge_pos: (x,y,z) of hinge point
    hinge_axis: (x,y,z) of hinge axis
    move_angle: total angle to move (unit: rad)
    step: number of interpolation steps
    """
    initial_offset = pose1[:3] - hinge_pos
    initial_rot = R.from_quat(pose1[3:], scalar_first=True)
    angles = np.linspace(0, move_angle, step)

    interpolated_poses = []
    for angle in angles:
        delta_rot = R.from_rotvec(angle * np_normalize_vector(hinge_axis))
        new_offset = delta_rot.apply(initial_offset)
        new_pos = hinge_pos + new_offset
        new_rot = delta_rot * initial_rot

        # Build new transformation matrix
        new_pose = np.zeros(7)
        new_pose[:3] = new_pos
        new_pose[3:] = new_rot.as_quat(scalar_first=True)
        interpolated_poses.append(new_pose)

    return interpolated_poses


def np_interp_qpos(qpos1: np.ndarray, qpos2: np.ndarray, step: int) -> np.ndarray:
    return np.linspace(qpos1, qpos2, step + 1)[1:]


def interp_move_traj(init_pose: np.ndarray, move_cfg: dict, step: int):
    if move_cfg["type"] == "slide":
        target_pose = np.copy(init_pose)
        target_pose[:3] += move_cfg["axis"] * move_cfg["distance"]
        move_pose_lst = np_interp_slide(init_pose, target_pose, step)
    elif move_cfg["type"] == "hinge":
        move_pose_lst = np_interp_hinge(
            pose1=init_pose,
            hinge_pos=move_cfg["pos"],
            hinge_axis=move_cfg["axis"],
            move_angle=move_cfg["distance"],
            step=step,
        )
    elif move_cfg["type"] != "force_closure":
        raise NotImplementedError(
            f"Unsupported task type: {move_cfg['type']}. Avaiable choices: 'hinge', 'slide', 'force_closure'."
        )
    return move_pose_lst
