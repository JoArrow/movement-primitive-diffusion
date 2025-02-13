#from omni.isaac.lab.app import AppLauncher

# Launch the simulation app
#app_launcher = AppLauncher(headless=True)
#simulation_app = app_launcher.app

from copy import deepcopy
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
from .examples.utils.furniture_utils import get_part_names
from .examples.aloha_env_cfg import AlohaEnvCfg


OPEN_THRESH = 1.0
CLOSE_THRESH = 0.0

START_ARM_POSE = [0.0, -0.96, 0.8, 0.0, -0.3, 0.0, 0.02239, -0.02239]
START_ARM_VEL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
                 dt = 0.005555556,
                 scaler_config: DictConfig = None,
                 time_limit: Optional[int] = None,
                 num_upload_successful_videos: int = 0,
                 num_upload_failed_videos: int = 0,
                 show_images: bool = False,
                 task = 'lamp'
                 ):
        
        env_cfg = AlohaEnvCfg(task='lamp', 
                              event_type='record',
                              dt = dt)  
        super().__init__(cfg=env_cfg)

        self.t_obs = t_obs
        self.time_limit = time_limit
        self.dt = dt
        
        self.observation_buffer = defaultdict(partial(deque, maxlen=self.t_obs))
        self.latest_action: np.ndarray = None
        self.latest_action_vel: np.ndarray = START_ARM_VEL[:7]
        self.time_step_counter: int = 0

        self.task = task

        # Get the robot articulations
        self.robot_left: Articulation = self.scene["robot_left"]  
        self.robot_right: Articulation = self.scene["robot_right"]

        # Create the robot controller
        self.controller_left = AlohaController(self, "robot_left")  
        self.controller_right = AlohaController(self, "robot_right")


    def reset(self, *args, **kwargs):
        # reset step counter
        self.time_step_counter = 0

        # Reset robot controller
        self.controller_left.reset()  
        self.controller_right.reset()

        # Beaming to start position and velocity
        self.controller_left.set_robot_joint_pos(torch.tensor(START_ARM_POSE), 
                                                 torch.tensor(START_ARM_VEL))
        self.controller_right.set_robot_joint_pos(torch.tensor(START_ARM_POSE),
                                                  torch.tensor(START_ARM_VEL))

        # reset environment
        env_obs, info = super().reset()

        # overwrite obs with custom function
        env_obs = self.get_observation()

        # Fill observation buffer with initial values
        for _ in range(self.t_obs):
            self.update_observation_buffer(env_obs)

        return env_obs, info



    def check_success(self, 
                      num_envs = 1,
                      task = 'lamp') -> bool | list[bool]:
        """
        Check if the furniture is assembled.
        Args:
            num_envs (int): The number of environments.
            task (str): The task being performed.
        Returns:
            bool or list: A boolean indicating whether the furniture is assembled in the single environment or a list of booleans indicating whether the furniture is assembled in each environment.
        """
        if num_envs == 1: # check if the furniture is assembled in the single environment
            part_poses = {part: self.scene[part].data.root_state_w[0][:7].cpu().numpy() for part in ['lamp_base', 'lamp_bulb', 'lamp_hood']}
            return furniture_assembly_check(task, part_poses) # returns a boolean
        
        else: # check for each environment if the furniture is assembled.
            part_poses = self.get_part_poses(num_envs=num_envs, task=task)
            return self.is_assembled(part_poses, task=task) # returns a list of booleans


    def get_part_poses(
            self,
            num_envs: int = 1,
            task: str = 'lamp') -> dict:
        """
        Get the poses of the specified parts in the environment.
        Args:
            task (str): The task being performed.
            part_names (list): A list of part names.
        Returns:
            dict: A dictionary containing the poses of the specified parts for all envs. The shape of each pose is (num_envs, 7).
        """
        part_names: list = get_part_names(task)
        part_poses = {} # dictionary to store the poses of the parts. Keys are the part names, Values are poses of shape (num_envs, 7). 
        for part in part_names:
            part_poses[part] =torch.cat(
                (self.scene[part].data.root_state_w[:,:3] - self.scene.env_origins[:,:3], 
                 self.scene[part].data.root_state_w[:,3:7]), dim=1)
            for key, val in part_poses.items():
                assert(val.shape[0] == num_envs)
                assert(val.shape[1] == 7)
        return part_poses


    def is_assembled(part_poses: dict,
                     task: str = 'lamp') -> List[bool]:
        """
        Check if the furniture is assembled.
        Args:
            part_poses (dict): A dictionary containing the poses of the parts. Keys are the part names, Values are poses of shape (num_envs, 7).
            num_envs (int): The number of environments.
            task (str): The task being performed.
        Returns:
            list: A list of booleans indicating whether the furniture is assembled in each environment.
        """
        result = []
        # get the number of environments from the shape of the poses of the parts
        num_envs = part_poses[0].shape[0]
        for i in range(num_envs):
            # get the poses of the parts for the i-th environment
            poses_i = {key: values[i] for key, values in part_poses.items()} 
            # check if the furniture is assembled in the i-th environment
            result.append(furniture_assembly_check(task = task,  poses=poses_i)) 
        assert(len(result) == num_envs)
        return result


    def step(self, action: np.ndarray):
        self.time_step_counter += 1


        # update the latest action and action velocity
        if self.latest_action is not None:
            self.latest_action_vel = (action - self.latest_action) / self.dt
        self.latest_action = deepcopy(action)


        # apply action to environment
        action_l = action[:7]
        action_r = action[7:]

        self.controller_left.move_forward_kinematics(action_l)
        self.controller_right.move_forward_kinematics(action_r)

        _, info = super().step(torch.cat((self.robot_left.data.joint_pos_target, 
                                          self.robot_right.data.joint_pos_target), 1))


        # get observation and update observation buffer
        env_obs = self.get_observation()
        self.update_observation_buffer(env_obs)


        # TODO: We do not use rewards, so we just set it to 0.0 
        reward = 0.0 


        # Terminate if the furniture is assembled or the time limit is reached
        terminated = self.check_success()
        truncated = False
        if self.time_limit and self.time_step_counter > self.time_limit:
            terminated = True
            truncated = True


        # Create emtpy info
        #info: dict[str, Any] = {}

        # assert(len(env_obs) == len(self.observation_buffer))
        # assert(env_obs.keys() == self.observation_buffer.keys())

        return env_obs, reward, terminated, truncated, info



    # Get observation dictionaries from the environment
    def get_observation_dict(self, add_batch_dim: bool = True) -> dict[str, torch.Tensor]:
        """
        Returns the observation buffer as a dictionary of torch tensors.
        """
        return deque_to_array(self.observation_buffer, add_batch_dim=add_batch_dim)



    def update_observation_buffer(self, env_obs: dict[str, torch.Tensor]) -> None:
        """
        Updates the observation buffer with the latest observations, agent state, and action information.
        """
        for key, val in env_obs.items():
            self.observation_buffer[key].append(val)
        
        # # Assert that each value of observation_buffer contains the same number of elements
        # num_elements = len(self.observation_buffer[key])
        # for key, val in self.observation_buffer.items():
        #     assert len(val) == num_elements



    def get_observation(self) -> dict[str, torch.Tensor]:
        """
        Returns the current observation as a dictionary of torch tensors.
        """
        env_obs: dict[str, torch.Tensor] = {}

        for part in ['lamp_base', 'lamp_bulb', 'lamp_hood']:
            env_obs[part] = torch.cat((self.scene[part].data.root_state_w[:,:3] - self.scene.env_origins[:,:3], 
                                       self.scene[part].data.root_state_w[:,3:7]), dim=1).cpu().squeeze()
        
        # agent state. Position and velocity of the agent
        env_obs["agent_pos_l"] = self.scene['robot_left'].data.joint_pos.cpu().squeeze()
        env_obs["agent_pos_r"] = self.scene['robot_right'].data.joint_pos.cpu().squeeze()

        env_obs["agent_vel_l"] = self.scene['robot_left'].data.joint_vel.cpu().squeeze()
        env_obs["agent_vel_r"] = self.scene['robot_right'].data.joint_vel.cpu().squeeze()

        if self.latest_action is None:
            # if the latest action is None, we set it to the agent position and set the action velocity to the agent velocity
            # only copy the first 7 values, the last value is not used for actions (because gripper fingers act the same)
            env_obs["action_l"] = env_obs["agent_pos_l"][:7].numpy()
            env_obs["action_r"] = env_obs["agent_pos_r"][:7].numpy()
            env_obs["action_vel_l"] = env_obs["agent_vel_l"][:7].numpy()
            env_obs["action_vel_r"] = env_obs["agent_vel_r"][:7].numpy()
        else:
            env_obs["action_l"] = self.latest_action[:7]
            env_obs["action_r"] = self.latest_action[7:]
            env_obs["action_vel_l"] = self.latest_action_vel[:7]
            env_obs["action_vel_r"] = self.latest_action_vel[7:]

        return env_obs    