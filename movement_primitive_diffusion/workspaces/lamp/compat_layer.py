# from omni.isaac.lab.app import AppLauncher
# Launch the simulation app
# app_launcher = AppLauncher(headless=True)
# simulation_app = app_launcher.app

from typing import Any, Dict, List, Tuple, Optional, Union
from collections import deque, defaultdict
from functools import partial
from omegaconf import DictConfig
import torch
import sys, os


from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.assets import Articulation

from movement_primitive_diffusion.utils.helper import deque_to_array

from .examples.utils.furniture_utils import furniture_assembly_check
from .examples.aloha_env_cfg import AlohaEnvCfg


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
        self.time_step_counter: int = 0


    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.time_step_counter = 0

        # Fill observation buffer with initial values
        for _ in range(self.t_obs):
            self.update_observation_buffer()

        return obs, info


    def step(self, action):
        super().step(action) # call actual step function
        self.update_observation_buffer() # update observation buffer after each step
        env_obs = {key: torch.tensor(value[-1]) for key, value in self.observation_buffer.items()}

        # TODO: We do not use rewards, so is this the right way to simply set it to 0.0?
        reward = 0.0 

        # Terminate if the furniture is assembled or the time limit is reached
        part_poses = {part: self.scene[part].data.root_state_w[0][:7].cpu().numpy() for part in ['lamp_base', 'lamp_bulb', 'lamp_hood']}
        terminated = furniture_assembly_check('lamp', part_poses)
        self.time_step_counter += 1
        truncated = False
        if self.time_limit is not None and self.scene.time > self.time_limit:
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
        # observation_buffer values should be tensor of shape (1, t_obs, feature_size)

        # lamp parts
        for part in ['lamp_base', 'lamp_bulb', 'lamp_hood']:   
            self.observation_buffer[part].append(torch.cat((self.scene[part].data.root_state_w[:, :3].cpu() - self.scene.env_origins[:, :3].cpu(),
                                                            self.scene[part].data.root_state_w[:, 3:7].cpu()), dim=1).squeeze())

        # agent state. Position and velocity of the agent
        self.observation_buffer["agent_pos_l"].append(self.scene['robot_left'].data.joint_pos.cpu().squeeze())
        self.observation_buffer["agent_pos_r"].append(self.scene['robot_right'].data.joint_pos.cpu().squeeze())

        self.observation_buffer["agent_vel_l"].append(self.scene['robot_left'].data.joint_vel.cpu().squeeze())
        self.observation_buffer["agent_vel_r"].append(self.scene['robot_right'].data.joint_vel.cpu().squeeze())

        # Add action_l and action_r to the observation buffer.
        self.observation_buffer["action_l"].append(self.scene['robot_left'].data.joint_pos_target.cpu().squeeze())
        self.observation_buffer["action_r"].append(self.scene['robot_right'].data.joint_pos_target.cpu().squeeze())

        # TODO: I though this is calculated by the model?
        self.observation_buffer["action_vel_l"].append(self.scene['robot_left'].data.joint_vel_target.cpu().squeeze())
        self.observation_buffer["action_vel_r"].append(self.scene['robot_right'].data.joint_vel_target.cpu().squeeze())




#if __name__ == "__main__":
#    # Initialize the environment
#    t_obs = 10
#    time_limit = 1000
#    env = AlohaLampEnv(t_obs=t_obs, time_limit=time_limit)

    # Example of running the environment for a few steps
#    for _ in range(5):
#        action = torch.zeros(env.action_space.shape)  # Assuming action space is known and is a tensor
#        obs, reward, terminated, truncated, info = env.step(action)
#        print(f"Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
#        if terminated:
#            break