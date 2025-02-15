from movement_primitive_diffusion.workspaces.lamp.compat_layer import AlohaLampEnv
from omegaconf import DictConfig
from typing import Dict, List, Optional
import hydra
import numpy as np
import math
import torch
import wandb
import time

from omegaconf import DictConfig
from tqdm import tqdm
from typing import Dict, List, Optional
from pathlib import Path

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.utils.gym_utils.async_vector_env import AsyncVectorEnv
from movement_primitive_diffusion.utils.gym_utils.sync_vector_env import SyncVectorEnv
from movement_primitive_diffusion.utils.helper import list_of_dicts_of_arrays_to_dict_of_arrays
from movement_primitive_diffusion.utils.video import save_video_from_array
from movement_primitive_diffusion.utils.visualization import tile_images
from movement_primitive_diffusion.workspaces.base_vector_workspace import BaseVectorWorkspace

from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.envs import ManagerBasedEnv
from aloha_env_cfg import AlohaEnvCfg



class AlohaLampWorkspace():
    def __init__(
        self,
        env_config: DictConfig,
        t_act: int,
        num_parallel_envs: int,
        shared_memory: bool = False,
        num_upload_successful_videos: int = 5,
        num_upload_failed_videos: int = 5,
        video_dt: float = 0.1,
        show_images: bool = False,
        annotate_videos: bool = False,
        timeout: Optional[float] = None,
    ):
        super().__init__(
            env_config = env_config,
            t_act = t_act,
            num_parallel_envs = num_parallel_envs,
            shared_memory = shared_memory,
            async_vector_env = False,
            num_upload_successful_videos = num_upload_successful_videos,
            num_upload_failed_videos = num_upload_failed_videos,
            video_dt = video_dt,
            show_images = show_images,
            annotate_videos = annotate_videos,
            timeout = timeout, # TODO: remove this parameter
        )

        # Launch the simulation app
        app_launcher = AppLauncher()#(headless=True)
        simulation_app = app_launcher.app

        
        self.env = hydra.utils.instantiate(env_config)
        self.t_act = t_act
        self.num_upload_successful_videos = num_upload_successful_videos
        self.num_upload_failed_videos = num_upload_failed_videos
        self.show_images = show_images
        self.time_limit = env_config.time_limit
        
        super().__init__(env_config, 
                         t_act,
                         num_upload_successful_videos,
                         num_upload_failed_videos,
                         show_images)


    
    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        """
        return:
            dict: {"success_rate": float,}
        """
        self.num_successful_trajectories = 0
        self.num_failed_trajectories = 0
        frames_of_successful_trajectories = []
        frames_of_failed_trajectories = []

        # How ofter we call agent.predict in total
        self.n_predict_calls = math.ceil(self.time_limit / self.t_act)

        # Variables to keep track of the env's done and successful status, and rendered frames
        done_buffer = [False] * self.num_parallel_envs
        successful_buffer = [False] * self.num_parallel_envs
        frame_buffer = [[] for _ in range(self.num_parallel_envs)]

        done: bool = False

        self.reset_env(caller_locals=locals())


        for i in (pbar := tqdm(range(self.n_predict_calls), desc="Testing agent", leave=False)):
            observation_buffer = self.env.get_observation_dict()

            # Create torch tensors from numpy arrays
            # TODO: make sure that the values (tensors) of observation_buffer have shape (num_env, t_obs, ...)
            for key, val in observation_buffer.items():
                observation_buffer[key] = torch.from_numpy(val)

            # Process the observation buffer to get observations and extra inputs
            observation, extra_inputs = agent.process_batch.process_env_observation(observation_buffer)

            # Move observations and extra inputs to device
            for key, val in observation.items():
                observation[key] = val.to(agent.device)
            for key, val in extra_inputs.items():
                if isinstance(val, torch.Tensor):
                    extra_inputs[key] = val.to(agent.device)

            # Predict the next action sequence
            actions = agent.predict(observation, extra_inputs)

            # Remove batch dimension and move to cpu and get numpy array
            actions = actions.squeeze(0).cpu().numpy()
            assert actions.ndim == 3 and actions.shape[0] == self.num_parallel_envs, f"Actions should be of shape (B, T, N) for B parallel environments, T timesteps, and N action dimensions. Got shape {actions.shape}"


            # Execute at most t_act actions of the sequence in the vectorized environment
            for action_step_index in range(self.t_act):
                # Execute the actions in the vectorized environment and get the rendered frames
                # TODO: Check what format action should have, to pass action to every environment.
                env_obs, env_reward, env_terminated, env_truncated, env_info = self.env.step(actions[:, action_step_index, :])

                #post_step_frames = self.render_function(caller_locals=locals())

                # Update buffer for done and successful environments, and add frames to the frame buffer
                done_buffer = env_terminated | env_truncated # TODO: in our case all environments are terminated at the same time. maybe we can change that.
                successful_buffer = self.check_success_hook(locals())

                # If all environments are done before max_action_sequences are executed, do not execute the remaining action sequences
                if all(done_buffer):
                    break
        
        self.post_step_hook(caller_locals=locals())

        pbar.set_postfix(success_rate=sum(successful_buffer) / len(successful_buffer))

    def post_step_hook(self, caller_locals: Dict) -> None:
        """Function to modify post step behavior in subclasses.

        Updating the current best value of some metric.
        """
        pass

    def check_success_hook(self, caller_locals: dict) -> bool:
        """Function to modify success check behavior in subclasses.

        For example for checking if a certain part is assembled.

        """
        return self.env.check_success()

    def reset_env(self, caller_locals: dict) -> np.ndarray:
        """Function to modify reset behavior in subclasses.

        For example for setting a random seed, or passing an options dict.

        """
        return self.env.reset()
    
