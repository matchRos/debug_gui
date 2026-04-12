from typing import Dict

import rospy
from geometry_msgs.msg import PoseStamped

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class DescendToGraspStep(BaseStep):
    name = "descend_to_grasp"
    description = "Move from pre-grasp to grasp height above the table."

    def __init__(self):
        super().__init__()

        if not rospy.core.is_initialized():
            rospy.init_node("debug_gui_descend_to_grasp", anonymous=True)

        self.pub_left = rospy.Publisher(
            "/yumi/robl/cartesian_pose_command",
            PoseStamped,
            queue_size=1,
        )
        self.pub_right = rospy.Publisher(
            "/yumi/robr/cartesian_pose_command",
            PoseStamped,
            queue_size=1,
        )

    def run(self, state: PipelineState) -> Dict[str, object]:
        if not hasattr(state, "grasp_poses"):
            raise RuntimeError("No grasp poses available.")

        pose = state.pregrasp_poses[0]
        pos = pose["position"]
        rot = pose["rotation"]
        arm = pose.get("arm", "right")

        # target: 7 cm above table plane
        target_z = 0.07

        from scipy.spatial.transform import Rotation as R

        quat = R.from_matrix(rot).as_quat()

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"

        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(target_z)

        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        if arm == "left":
            self.pub_left.publish(msg)
        else:
            self.pub_right.publish(msg)

        rospy.sleep(2.0)

        return {
            "descend_sent": True,
            "arm": arm,
            "position": [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
        }
