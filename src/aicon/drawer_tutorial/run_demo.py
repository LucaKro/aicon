import time
from pathlib import Path

import lovely_tensors as lt
import numpy as np
import robosuite as suite
import torch
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action
from robosuite.utils.transform_utils import quat2mat
from robosuite.wrappers import VisualizationWrapper

from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion

# Import our custom environment
from aicon.drawer_tutorial.robosuite_drawer_env import DrawerOpenEnv
from aicon.middleware.python_sequential import build_components, run_component_sequence

# Initial Panda joint configurations. "default" is the pose this demo started with.
POSES = {
    "default": [-0.2, 0.2, 0.1, -2.0, 0.0, 1.5, 0.7],
    "right": [0.2, 0.2, 0.1, -2.0, 0.0, 1.5, 0.7],
}

# The drawer joint travels 4 cm, so _check_success (qpos < -0.1) never fires.
OPENED = -0.039


class Run:
    """Parameters for one run. Subclass and override what differs."""
    pose = "default"
    required_signs = True   # False clears EEDrawerGraspedConnection's required_signs_dict
    cabinet_yaw = 0.0       # degrees; rotates the cabinet, which is otherwise fixed
    steps = None            # None runs until interrupted
    seed = 0


class Default(Run):
    """The original demo. Opens at step 209."""
    steps = 300


class Right(Default):
    """Arm starts turned to the right. Opens at step 179."""
    pose = "right"


class RightNoSigns(Right):
    """Same start without the signs: the hand never closes and the drawer never moves.
    Here the signs are necessary."""
    required_signs = False


class RotatedCabinet(Default):
    """Cabinet turned -20 deg. Here the signs are what PREVENT the drawer opening:
    they suppress a release-and-regrasp that the unforced gradient would trigger."""
    cabinet_yaw = -20.0


class RotatedCabinetNoSigns(RotatedCabinet):
    """Same scene without the signs: releases at step 105, re-grips at 119, opens at 269."""
    required_signs = False


EXPERIMENTS = [Default, Right, RightNoSigns, RotatedCabinet, RotatedCabinetNoSigns]


def setup_env(device_type, initial_qpos=None):
    device = Keyboard(pos_sensitivity=1, rot_sensitivity=1)
    # Create our custom environment
    env = DrawerOpenEnv(
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        render_camera="agentview",
        horizon=100,
        control_freq=30,
        controller_configs=suite.load_controller_config(default_controller="OSC_POSITION"),
        initial_qpos=initial_qpos,
    )

    env = VisualizationWrapper(env)
    env.reset()
    device.start_control()
    return env, device


def main(device, env, rec_save_path=None, cfg=Default):
    robot = env.robots[0]
    device.start_control()

    # Print debug information about the drawer
    env.env.print_debug_info()

    # Get the observation using the original method
    env_obs = env._get_observations()

    # Access cabinet information in the same way as the original environment
    cabinet_position = env_obs["CabinetObject_pos"]
    cabinet_orientation = quat2mat(env_obs["CabinetObject_quat"])

    print(f"\nCabinet position from obs: {cabinet_position}")
    print(f"Cabinet orientation matrix from obs:\n{cabinet_orientation}")

    # setup aicon
    component_building_functions, connection_building_functions, frame_rates = (
        get_building_functions_basic_drawer_motion(env)
    )
    components = build_components(component_building_functions)
    if not cfg.required_signs:
        for component in components.values():
            for connection in component.connections.values():
                connection.required_signs_dict = {}
    gripper_component = components["GripperAction"]
    gripper_velo = components["EEVelocities"]

    # Start sim
    curr_t = 0
    step, opened_at, qpos = 0, None, 0.0
    while cfg.steps is None or step < cfg.steps:
        action, grasp = input2action(
            device=device, robot=robot, active_arm="right", env_configuration="single-arm-opposed"
        )
        run_component_sequence(components, torch.tensor(curr_t))
        curr_commanded_vel = gripper_velo.quantities["action_velo_ee"]
        curr_commanded_gripper = gripper_component.quantities["gripper_activation"]
        action = np.concatenate(
            [curr_commanded_vel.cpu().numpy(), [2 * curr_commanded_gripper.squeeze().cpu().numpy() - 1]]
        )

        obs, rew, done, info = env.step(action)

        # the env's own success check tests qpos < -0.1, which can never happen
        qpos = float(env.env.sim.data.qpos[env.env.cabinet_qpos_addrs])
        if opened_at is None and qpos <= OPENED:
            opened_at = step
            print(f"\n*** OPENED at step {step} (qpos={qpos:.4f}) ***\n")

        curr_t += env.control_timestep
        env.render()
        step += 1

    print(f"SUMMARY {cfg.__name__}: opened_at={opened_at} final_qpos={qpos:.4f}\n")
    return opened_at


def run_demo(cfg=Default):
    torch.random.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"=== {cfg.__name__}: pose={cfg.pose} required_signs={cfg.required_signs} ===")

    # Pass the initial pose to the setup function
    env, device = setup_env("keyboard", initial_qpos=np.array(POSES[cfg.pose]))
    if cfg.cabinet_yaw:
        half = np.radians(cfg.cabinet_yaw) / 2.0
        env.env.sim.model.body_quat[env.env.cabinet_object_id] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        env.env.sim.forward()
    return main(device, env, cfg=cfg)


if __name__ == "__main__":
    # one class, or e.g. EXPERIMENTS to run them all in sequence
    for case in EXPERIMENTS: #[Default]:
        run_demo(case)
