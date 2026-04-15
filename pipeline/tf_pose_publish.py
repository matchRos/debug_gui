"""
Transform Cartesian poses from the internal planning frame (yumi_base_link) to the
frame expected by the motion stack (typically ``world``), using tf2.

``lookup_transform`` is used only to fetch translation + quaternion. Pose composition
is done in NumPy/SciPy to avoid ``tf2_geometry_msgs`` / ``Buffer.transform`` Python
paths that have triggered heap issues (e.g. ``corrupted size vs. prev_size``).
"""

from __future__ import annotations

import threading

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as SciRotation

_tf_buffer = None
_tf_listener = None
_tf_lock = threading.Lock()


def ensure_tf_buffer():
    """Lazy-init tf2 buffer + listener (requires rospy.init)."""
    global _tf_buffer, _tf_listener
    with _tf_lock:
        if _tf_buffer is None:
            _tf_buffer = tf2_ros.Buffer(rospy.Duration(60.0))
            _tf_listener = tf2_ros.TransformListener(_tf_buffer)
        return _tf_buffer


def _apply_tf_to_pose_stamped(
    msg_in: PoseStamped, tf_msg, target_frame: str
) -> PoseStamped:
    """
    Apply ``geometry_msgs/TransformStamped`` from ``lookup_transform(target, source)``:
    maps points/orientations from ``source`` to ``target``.
    """
    tr = tf_msg.transform.translation
    qr = tf_msg.transform.rotation
    t = np.array([tr.x, tr.y, tr.z], dtype=float)
    r_tf = SciRotation.from_quat([qr.x, qr.y, qr.z, qr.w])

    p = msg_in.pose.position
    p_src = np.array([p.x, p.y, p.z], dtype=float)
    p_tgt = r_tf.apply(p_src) + t

    qp = msg_in.pose.orientation
    r_pose = SciRotation.from_quat([qp.x, qp.y, qp.z, qp.w])
    r_out = r_tf * r_pose
    q = r_out.as_quat()

    out = PoseStamped()
    out.header.stamp = rospy.Time.now()
    # Must match the requested target frame (caller passes it explicitly).
    out.header.frame_id = target_frame
    out.pose.position.x = float(p_tgt[0])
    out.pose.position.y = float(p_tgt[1])
    out.pose.position.z = float(p_tgt[2])
    out.pose.orientation.x = float(q[0])
    out.pose.orientation.y = float(q[1])
    out.pose.orientation.z = float(q[2])
    out.pose.orientation.w = float(q[3])
    return out


def transform_pose_stamped_to_frame(
    msg_in: PoseStamped,
    target_frame: str,
    timeout_sec: float = 2.0,
) -> PoseStamped:
    """
    Transform ``msg_in`` (pose expressed in ``msg_in.header.frame_id``) into
    ``target_frame``.
    """
    buf = ensure_tf_buffer()
    src = msg_in.header.frame_id
    if src == target_frame:
        out = PoseStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = target_frame
        out.pose = msg_in.pose
        return out

    timeout = rospy.Duration(float(timeout_sec))
    try:
        with _tf_lock:
            tf_msg = buf.lookup_transform(
                target_frame, src, rospy.Time(0), timeout=timeout
            )
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ) as e:
        raise RuntimeError(
            f"tf2 could not look up {src!r} -> {target_frame!r}: {e}"
        ) from e
    return _apply_tf_to_pose_stamped(msg_in, tf_msg, target_frame)
