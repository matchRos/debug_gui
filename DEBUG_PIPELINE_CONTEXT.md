# Project Context: YuMi Cable Routing Debug Pipeline

## Goal
This repository contains a debug-oriented reimplementation of parts of the original cable routing pipeline for a dual-arm ABB YuMi robot.

The purpose is:
- understand the original pipeline step by step
- isolate bugs more easily
- test new motion logic safely in a modular GUI pipeline
- gradually reconstruct the original routing logic in a simpler, debuggable form

This debug pipeline is intentionally independent from the original monolithic implementation.

---

## High-level architecture

### GUI
There is a PyQt-based debug GUI with:
- a list of pipeline steps
- logs
- an image/overlay panel
- controls to run the next step or a selected step

The GUI should always make intermediate results visible if possible.

### Pipeline pattern
Each pipeline step is implemented as its own class:
- one class per step
- shared runtime data in `PipelineState`
- logic should be kept modular and readable
- small steps are preferred over large monolithic functions

### Important design principle
When extending the debug pipeline:
- preserve transparency
- preserve debuggability
- avoid hiding important state transitions inside giant functions
- prefer storing intermediate results explicitly in `PipelineState`

---

## Current debug pipeline concept

Typical steps currently include:
- home_arms
- init_environment
- prepare_routing
- trace_cable
- trace_to_world
- compute_orientation
- grasp_planning
- grasp_pose
- visualize_grasps
- pregrasp_pose
- robot_motion
- optional wrist unwind step
- descend_to_grasp
- close gripper steps
- lift_after_grasp
- plan_first_route

Not every step must exactly match the original pipeline yet.

---

## Robot setup

### Robot
- ABB YuMi
- dual-arm setup
- ROS-based control

### Relevant arm topics/services
Examples currently used in the debug pipeline:
- `/yumi/robl/cartesian_pose_command`
- `/yumi/robr/cartesian_pose_command`
- `/yumi/robl/slowly_approach_pose`
- `/yumi/robr/slowly_approach_pose`
- `/yumi/gripper_l/open`
- `/yumi/gripper_r/open`
- `/yumi/gripper_l/close`
- `/yumi/gripper_r/close`
- `/yumi/home_both_arms`
- `/yumi/joint_group_velocity_command`
- `/joint_states`

### Joint naming
The `/joint_states` order is:

Left arm:
- yumi_robl_joint_1
- yumi_robl_joint_2
- yumi_robl_joint_3
- yumi_robl_joint_4
- yumi_robl_joint_5
- yumi_robl_joint_6
- yumi_robl_joint_7

Right arm:
- yumi_robr_joint_1
- yumi_robr_joint_2
- yumi_robr_joint_3
- yumi_robr_joint_4
- yumi_robr_joint_5
- yumi_robr_joint_6
- yumi_robr_joint_7

Then:
- gripper_l_joint
- gripper_r_joint

### Important current practical behavior
- Cartesian target poses do not reliably control which q7 branch the IK/controller chooses
- therefore, wrist unwinding may need to be handled as a separate step using joint velocity commands
- for problematic wrist configurations, a separate unwind step is preferred over forcing everything through one Cartesian pose

---

## Camera and geometry

### Camera
- ZED camera via ROS
- image tracing happens in pixel space
- path is later projected into world coordinates

### Extrinsics
The debug environment loads camera-to-base transforms into:
- `state.env.T_CAM_BASE["left"]`
- `state.env.T_CAM_BASE["right"]`

### World/path assumptions
- cable path starts near the first routing object / cable start
- ordering along the cable is important
- arc-length order matters more than Euclidean distance
- current table plane assumption: cable plane is approximately world z = constant
- current grasp/table safety height is based on a simple flat table model

---

## Grasping logic

### Grasp planning
Dual-arm grasping is based on two points along the cable path.
Important:
- the grasps are ordered along cable path, not by Euclidean proximity
- `path_index` is used to preserve cable order
- larger `path_index` means farther from cable start

### Arm assignment
A grasp pose gets assigned to either:
- `"left"`
- `"right"`

This assignment must remain explicit in pose dictionaries.

### Pose dictionaries
Typical pose dictionaries may contain:
- `position`
- `rotation`
- `approach_axis`
- `arm`
- `path_index`

If possible, preserve metadata like `path_index` through all later steps.

### Pregrasp
Pregrasp should usually be directly above grasp in world z.
For this project, vertical moves are preferred over tool-axis offsets when the goal is a clean vertical descend.

---

## Important motion constraints

### Descend/grasp order
A key physical issue is:
if the arm nearer to the cable start descends too early, it may press the cable down and cause the other arm to miss the cable.

Therefore:
- the arm farther from the cable start must descend first
- in some cases it should also close first
- sequential execution is preferred over naive simultaneous descend

### Lift
After grasping, both arms may move up together by a small z offset to avoid staying too close to the table.

### Wrist issues
If q7 gets into a poor branch:
- do not assume a 180° tool z rotation will enforce the desired sign of wrist rotation
- +180° and -180° are equivalent as final orientation
- branch handling may require either:
  - a non-symmetric orientation bias
  - or a separate q7 unwind step

---

## Current software architecture expectations

When modifying code:
- prefer minimal, local changes
- preserve readability
- preserve explicit debug outputs
- avoid introducing hidden side effects
- avoid rewriting unrelated files
- keep comments in English
- keep ROS code practical and robust
- include basic runtime checks if helpful

When proposing code:
- show complete functions or complete files when possible
- avoid partial patches that are hard to place
- mention where new files should live

---

## GUI expectations

The GUI should visualize intermediate state whenever possible.

Current useful overlay layers include:
- `state.rgb_image`
- `state.trace_overlay`
- `state.routing_overlay`
- `state.grasp_overlay`

If a new planning step is added, it should ideally draw:
- current clip(s)
- route direction
- current grasp/start point
- planned target point
- connection lines / arrows where helpful

Visualization is an important debugging tool in this project.

---

## Original pipeline relation

The debug pipeline is based on an original monolithic pipeline in:
- `cable_routing_new_pipeline.py`
- `env_new.py`

The original flow is roughly:
- initialize environment
- trace cable
- convert to world
- compute cable orientation
- grasp cable
- lift slightly
- repeatedly route around clips
- finish near final clip

The debug pipeline does not need to match this 1:1 immediately, but should move gradually toward the same functional behavior.

---

## How Cursor should help in this repo

Cursor should:
- understand the whole repo before making larger edits
- prefer small, testable steps
- preserve the current debug-pipeline architecture
- avoid collapsing everything into one large function
- keep state transitions explicit
- respect existing ROS topic/service interfaces
- be careful with robot safety and motion order
- not remove useful debug visualization

When uncertain:
- ask for the relevant file rather than guessing
- or propose the smallest safe refactor first