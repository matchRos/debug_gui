from typing import List, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from cable_routing.debug_gui.backend.planes import ensure_min_plane_height, get_routing_plane


def split_dual_arm_poses(poses):
    """
    Extract exactly one left and one right pose from a list of two pose dicts.
    Each pose dict must contain the key 'arm' with value 'left' or 'right'.
    """
    if len(poses) != 2:
        raise RuntimeError("Expected exactly 2 poses for dual-arm motion.")

    left_pose = None
    right_pose = None

    for pose in poses:
        arm = pose.get("arm", None)
        if arm == "left":
            left_pose = pose
        elif arm == "right":
            right_pose = pose

    if left_pose is None or right_pose is None:
        raise RuntimeError("Need exactly one left pose and one right pose.")

    return left_pose, right_pose


def pose_to_msg(position, rotation, frame_id="world"):
    """
    Convert a pose dict entry (position + rotation matrix) into a PoseStamped message.
    rotation must be a 3x3 rotation matrix.
    """
    pos = np.asarray(position, dtype=float).reshape(3)
    rot = np.asarray(rotation, dtype=float).reshape(3, 3)

    quat = R.from_matrix(rot).as_quat()  # x, y, z, w

    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id

    msg.pose.position.x = float(pos[0])
    msg.pose.position.y = float(pos[1])
    msg.pose.position.z = float(pos[2])

    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])

    return msg, quat


def compute_pose_distance(pose_a, pose_b):
    """
    Euclidean 3D distance between two pose dicts.
    """
    pos_a = np.asarray(pose_a["position"], dtype=float).reshape(3)
    pos_b = np.asarray(pose_b["position"], dtype=float).reshape(3)
    return float(np.linalg.norm(pos_a - pos_b))


def compute_xy_shift(pose_a, pose_b):
    """
    XY distance between two pose dicts.
    Useful to verify that descend motion is mostly vertical.
    """
    pos_a = np.asarray(pose_a["position"], dtype=float).reshape(3)
    pos_b = np.asarray(pose_b["position"], dtype=float).reshape(3)
    return float(np.linalg.norm(pos_a[:2] - pos_b[:2]))


def validate_min_distance(left_pose, right_pose, min_dist_xyz, label="Poses"):
    """
    Raise an error if the two end-effector targets are too close in 3D.
    """
    dist_xyz = compute_pose_distance(left_pose, right_pose)
    if dist_xyz < min_dist_xyz:
        raise RuntimeError(
            f"{label} too close: distance={dist_xyz:.3f} m < {min_dist_xyz:.3f} m"
        )
    return dist_xyz


def publish_dual_arm_targets(
    pub_left,
    pub_right,
    left_pose,
    right_pose,
    frame_id="world",
) -> Tuple[PoseStamped, np.ndarray, PoseStamped, np.ndarray]:
    """
    Publish both arm targets effectively simultaneously.
    Returns:
        left_msg, left_quat, right_msg, right_quat
    """
    left_msg, left_quat = pose_to_msg(
        left_pose["position"], left_pose["rotation"], frame_id
    )
    right_msg, right_quat = pose_to_msg(
        right_pose["position"], right_pose["rotation"], frame_id
    )

    now = rospy.Time.now()
    left_msg.header.stamp = now
    right_msg.header.stamp = now

    pub_left.publish(left_msg)
    pub_right.publish(right_msg)

    return left_msg, left_quat, right_msg, right_quat


def get_path_progress_for_pose(pose):
    """
    Return progress along the cable path.
    Higher value means farther away from the cable start.
    The pose must contain either:
      - 'path_s'
      - or 'path_index'
    """
    if "path_s" in pose:
        return float(pose["path_s"])
    if "path_index" in pose:
        return float(pose["path_index"])
    raise RuntimeError(
        "Pose needs either 'path_s' or 'path_index' to determine cable order."
    )


def order_poses_for_staggered_motion(left_pose, right_pose):
    """
    Determine which arm should move first.
    Rule:
      the arm farther from cable start moves first.
    """
    left_prog = get_path_progress_for_pose(left_pose)
    right_prog = get_path_progress_for_pose(right_pose)

    if left_prog > right_prog:
        first_arm = "left"
        first_pose = left_pose
        second_arm = "right"
        second_pose = right_pose
    else:
        first_arm = "right"
        first_pose = right_pose
        second_arm = "left"
        second_pose = left_pose

    return {
        "first_arm": first_arm,
        "first_pose": first_pose,
        "second_arm": second_arm,
        "second_pose": second_pose,
        "left_progress": left_prog,
        "right_progress": right_prog,
    }


def publish_staggered_dual_arm_targets(
    pub_left,
    pub_right,
    left_pose,
    right_pose,
    delay_s=0.20,
    frame_id="world",
):
    """
    Publish the two arm targets with a small delay.
    The arm farther from the cable start is sent first.
    Returns:
        left_msg, left_quat, right_msg, right_quat, order
    """
    order = order_poses_for_staggered_motion(left_pose, right_pose)

    left_msg, left_quat = pose_to_msg(
        left_pose["position"], left_pose["rotation"], frame_id
    )
    right_msg, right_quat = pose_to_msg(
        right_pose["position"], right_pose["rotation"], frame_id
    )

    now = rospy.Time.now()
    left_msg.header.stamp = now
    right_msg.header.stamp = now

    if order["first_arm"] == "left":
        pub_left.publish(left_msg)
        rospy.sleep(delay_s)
        pub_right.publish(right_msg)
    else:
        pub_right.publish(right_msg)
        rospy.sleep(delay_s)
        pub_left.publish(left_msg)

    return left_msg, left_quat, right_msg, right_quat, order


def msg_pose_to_dict(msg):
    """
    Extract xyz position from PoseStamped for debug output.
    """
    return [
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
    ]


def quat_to_list(quat):
    """
    Convert quaternion array to plain Python list for debug output.
    """
    return [
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    ]


def wait_until_robot_settled(
    topic: str = "/joint_states",
    timeout_sec: float = 45.0,
    poll_rate_hz: float = 30.0,
    still_time_sec: float = 0.35,
    position_delta_rad: float = 0.004,
    joint_indices: Optional[List[int]] = None,
) -> None:
    """
    Block until reported joint positions are stable (no significant motion).

    Uses consecutive /joint_states samples: if the max absolute change across
    selected joints stays below position_delta_rad for still_time_sec, the robot
    is treated as settled. If joint_indices is None, all joints in the message
    are considered.
    """
    latest: List[Optional[JointState]] = [None]
    serial = [0]

    def _cb(msg: JointState) -> None:
        latest[0] = msg
        serial[0] += 1

    sub = rospy.Subscriber(topic, JointState, _cb, queue_size=1)

    rate = rospy.Rate(poll_rate_hz)
    deadline = rospy.Time.now().to_sec() + timeout_sec

    prev_pos: Optional[np.ndarray] = None
    last_serial = -1
    stable_since: Optional[float] = None

    try:
        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() > deadline:
                raise RuntimeError(
                    f"Timeout ({timeout_sec}s) waiting for robot to settle on {topic}."
                )

            rate.sleep()

            msg = latest[0]
            if msg is None or not msg.position:
                continue

            pos_full = np.asarray(msg.position, dtype=float)
            if joint_indices is not None:
                pos = pos_full[joint_indices]
            else:
                pos = pos_full

            seq = serial[0]
            if seq == last_serial:
                continue
            last_serial = seq

            if prev_pos is None:
                prev_pos = pos.copy()
                continue

            delta = float(np.max(np.abs(pos - prev_pos)))
            prev_pos = pos.copy()
            now = rospy.Time.now().to_sec()

            if delta < position_delta_rad:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= still_time_sec:
                    return
            else:
                stable_since = None
    finally:
        sub.unregister()


def enforce_pose_min_height(
    pose: dict,
    state,
    min_height_above_plane_m: float,
    clip_id=None,
) -> dict:
    """
    Return a pose copy with position clamped to a minimum distance above
    the configured routing plane.
    """
    plane = get_routing_plane(state.config, clip_id=clip_id)
    out = dict(pose)
    out["position"] = ensure_min_plane_height(
        np.asarray(pose["position"], dtype=float).copy(),
        plane,
        float(min_height_above_plane_m),
    )
    return out
