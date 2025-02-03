from movement_primitive_diffusion.utils.helper import deque_to_array

import os, sys

# TODO: Why is this needed? If paths are not appended, different packages will not be found
# Add the Isaac SDK paths to the Python path
#isaac_sim_path = os.path.abspath('/home/i53/student/pfeil/.local/share/ov/pkg/isaac-sim-4.2.0')
#sys.path.append(os.path.join(isaac_sim_path, 'exts/omni.isaac.core'))
#sys.path.append(os.path.join(isaac_sim_path, 'exts/omni.isaac.lab'))
#sys.path.append(os.path.join(isaac_sim_path, 'kit/python/lib/python3.10/site-packages'))



from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.assets import Articulation
from typing import Any, Dict, List, Tuple, Optional, Union
from collections import deque, defaultdict
from functools import partial
import torch

from examples.utils.furniture_utils import furniture_assembly_check
from examples.aloha_env_cfg import AlohaEnvCfg


class AlohaLampEnv(ManagerBasedEnv):
    def __init__(self,
                 t_obs: int,
                 time_limit: Optional[int] = None):
        
        env_cfg = AlohaEnvCfg(task='lamp')  
        super().__init__(cfg=env_cfg)

        self.t_obs = t_obs
        self.time_limit = time_limit
        
        self.observation_buffer = defaultdict(partial(deque, maxlen=self.t_obs))
        self.time_step_counter: int = 0




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
        # lamp parts
        for part in ['lamp_base', 'lamp_bulb', 'lamp_hood']:   
            self.observation_buffer = torch.cat((self.scene[part].data.root_state_w[:,:3] - self.scene.env_origins[:,:3], 
                                                 self.scene[part].data.root_state_w[:,3:7]), dim=1)
        
        # agent state. Position and velocity of the agent
        self.observation_buffer["agent_pos_l"].append(self.scene['robot_left'].data.joint_pos)
        self.observation_buffer["agent_pos_r"].append(self.scene['robot_right'].data.joint_pos)

        self.observation_buffer["agent_vel_l"].append(self.scene['robot_left'].data.joint_vel)
        self.observation_buffer["agent_vel_r"].append(self.scene['robot_right'].data.joint_vel)

