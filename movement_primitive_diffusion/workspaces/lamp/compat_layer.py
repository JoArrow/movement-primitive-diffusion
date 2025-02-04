#from omni.isaac.lab.app import AppLauncher

# Launch the simulation app
#app_launcher = AppLauncher(headless=True)
#simulation_app = app_launcher.app

from typing import Any, Dict, List, Tuple, Optional, Union
from collections import deque, defaultdict
from functools import partial
from omegaconf import DictConfig
import torch
import numpy as np

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import subtract_frame_transforms

from movement_primitive_diffusion.utils.helper import deque_to_array

from .examples.aloha import BODY_JOINTS, EE_BODY, NUM_ENVS, RECORD
from .examples.utils.furniture_utils import furniture_assembly_check
from .examples.aloha_env_cfg import AlohaEnvCfg


OPEN_THRESH = 1.0
CLOSE_THRESH = 0.0


class AlohaController:
    GRIPPER_FORCE = 25 

    def __init__(self, env: ManagerBasedEnv, robot_name: str):
        # Initialize controller with environment and robot name
        self.env: ManagerBasedEnv = env
        self.robot: Articulation = env.scene[robot_name]
        # self.robot.write_joint_stiffness_to_sim(self.JOINT_STIFFNESS, list(range(6)))
        # self.robot.write_joint_damping_to_sim(self.JOINT_DAMPING, list(range(6)))
        self.robot_entity_cfg: SceneEntityCfg = SceneEntityCfg(robot_name, joint_names=BODY_JOINTS, body_names=EE_BODY)

        self.robot_entity_cfg.resolve(env.scene)
        self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        # self.recording = Recording(BODY_JOINTS+GRIPPER_JOINTS)

    def reset(self):
        # Reset controller state
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        self.ee_pos_b, self.ee_quat_b = subtract_frame_transforms(self.robot.data.root_pos_w[0], self.robot.data.root_quat_w[0], ee_pose_w[0][0:3], ee_pose_w[0][3:7])
    
    def move_forward_kinematics(self, pose):
        # Normalize
        gripper = min(pose[-1], OPEN_THRESH)
        gripper = max(pose[-1], CLOSE_THRESH)
        gripper = gripper+CLOSE_THRESH
        gripper = gripper/(OPEN_THRESH+CLOSE_THRESH)

        gripper_both = 0
        # [AlohaController.GRIPPER_FORCE * (1 if self.gripper_cmd else 0) for _ in range(2)]
        # gripper = pose[-1]
        if(gripper < 0.5):
            gripper_both = AlohaController.GRIPPER_FORCE
            gripper_left = -AlohaController.GRIPPER_FORCE/2
            gripper_right = -AlohaController.GRIPPER_FORCE/2
        else:
            gripper_left = AlohaController.GRIPPER_FORCE/2
            gripper_right = AlohaController.GRIPPER_FORCE/2

        joints = np.append(pose[:-1], [gripper_left, gripper_right])
        joints = np.tile(joints, (NUM_ENVS, 1))    

        self.robot.set_joint_position_target(torch.Tensor(joints))
        self.robot.write_data_to_sim()

    def set_robot_joint_pos(self, 
                            joint_pos: torch.Tensor, 
                            joint_velocity: torch.Tensor = None):
        
        joint_pos = np.tile(joint_pos, (NUM_ENVS, 1))
        if joint_velocity is not None:
            joint_velocity = np.tile(joint_velocity, (NUM_ENVS, 1))


        self.robot.write_joint_state_to_sim(torch.Tensor(joint_pos), 
                                            torch.Tensor(joint_velocity) if joint_velocity is not None 
                                            else torch.zeros_like(torch.Tensor(joint_pos)))
        
        # assert that the joint pos and velocity are set correctly
        assert((self.robot.data.joint_pos.cpu().numpy() == joint_pos).all())
        assert((self.robot.data.joint_vel.cpu().numpy() == joint_velocity).all())



class AlohaLampEnv(ManagerBasedEnv):
    def __init__(self,
                 t_obs: int,
                 scaler_config: DictConfig = None,
                 time_limit: Optional[int] = None,
                 num_upload_successful_videos: int = 0,
                 num_upload_failed_videos: int = 0,
                 show_images: bool = False,):
        
        env_cfg = AlohaEnvCfg(task='lamp')  
        super().__init__(cfg=env_cfg)

        self.t_obs = t_obs
        self.time_limit = time_limit
        
        self.observation_buffer = defaultdict(partial(deque, maxlen=self.t_obs))
        self.latest_action = None
        self.latest_action_vel = None
        self.time_step_counter: int = 0

        self.reset() # reset the environment

        self.robot_left: Articulation = self.scene["robot_left"]  
        self.robot_right: Articulation = self.scene["robot_right"]

        # Create the robot controller
        self.controller_left = AlohaController(self, "robot_left")  
        self.controller_right = AlohaController(self, "robot_right")

        # Reset robot controller
        self.controller_left.reset()  
        self.controller_right.reset()

        # TODO: set the robot arms in start position?

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.time_step_counter = 0

        # Fill observation buffer with initial values
        for _ in range(self.t_obs):
            self.update_observation_buffer()

        return obs, info

    def step(self, action):
        self.latest_action = action

        action_l = action[:7]
        action_r = action[7:]
        assert(len(action_l) == len(action_r) == 7)

        self.controller_left.move_forward_kinematics(action_l)
        self.controller_right.move_forward_kinematics(action_r)

        super().step(torch.cat((self.robot_left.data.joint_pos_target, 
                                self.robot_right.data.joint_pos_target), 1))

        self.update_observation_buffer() # update observation buffer after each step

        env_obs = {key: torch.tensor(value[-1]) for key, value in self.observation_buffer.items()}

        reward = 0.0 # We do not use rewards, so we just set it to 0.0 

        # Terminate if the furniture is assembled or the time limit is reached
        part_poses = {part: self.scene[part].data.root_state_w[0][:7].cpu().numpy() for part in ['lamp_base', 'lamp_bulb', 'lamp_hood']}
        terminated = furniture_assembly_check('lamp', part_poses)
        self.time_step_counter += 1
        truncated = False
        if self.time_limit and self.time_step_counter > self.time_limit:
            terminated = True
            truncated = True

        info: dict[str, Any] = {}

        return env_obs, reward, terminated, truncated, info



    # Get observation dictionaries from the environment
    def get_observation_dict(self, add_batch_dim: bool = True) -> dict[str, torch.Tensor]:
        """
        Returns the observation buffer as a dictionary of torch tensors.
        """
        return deque_to_array(self.observation_buffer, add_batch_dim=add_batch_dim)


    def update_observation_buffer(self) -> None:
        """
        Updates the observation buffer with the latest observations, agent state, and action information.
        """
        # lamp parts
        for part in ['lamp_base', 'lamp_bulb', 'lamp_hood']:   
            self.observation_buffer[part].append(torch.cat((self.scene[part].data.root_state_w[:,:3] - self.scene.env_origins[:,:3], 
                                                 self.scene[part].data.root_state_w[:,3:7]), dim=1).cpu().squeeze())
        
        # agent state. Position and velocity of the agent
        self.observation_buffer["agent_pos_l"].append(self.scene['robot_left'].data.joint_pos.cpu().squeeze())
        self.observation_buffer["agent_pos_r"].append(self.scene['robot_right'].data.joint_pos.cpu().squeeze())

        self.observation_buffer["agent_vel_l"].append(self.scene['robot_left'].data.joint_vel.cpu().squeeze())
        self.observation_buffer["agent_vel_r"].append(self.scene['robot_right'].data.joint_vel.cpu().squeeze())


        if self.latest_action is not None:
            self.observation_buffer["action_l"].append(self.latest_action[:7])
            self.observation_buffer["action_r"].append(self.latest_action[7:])
        else:
            self.observation_buffer["action_l"].append(torch.zeros(7))
            self.observation_buffer["action_r"].append(torch.zeros(7))
        

        if self.latest_action_vel is not None:
            self.observation_buffer["action_vel_l"].append(self.latest_action_vel[:7])
            self.observation_buffer["action_vel_r"].append(self.latest_action_vel[7:])
        else:
            self.observation_buffer["action_vel_l"].append(torch.zeros(7))
            self.observation_buffer["action_vel_r"].append(torch.zeros(7))