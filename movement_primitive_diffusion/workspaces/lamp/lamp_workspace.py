from omegaconf import DictConfig
import hydra

from movement_primitive_diffusion.workspaces.base_workspace import BaseWorkspace

from omni.isaac.lab.app import AppLauncher
import torch
import cv2
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import Dict, List
from movement_primitive_diffusion.agents.base_agent import BaseAgent


class AlohaLampWorkspace(BaseWorkspace):
    def __init__(
        self,
        env_config: DictConfig,
        t_act: int,
        num_upload_successful_videos: int = 0,
        num_upload_failed_videos: int = 0,
        show_images: bool = False,
        ):
        

        # Launch the simulation app
        app_launcher = AppLauncher(headless=True)
        simulation_app = app_launcher.app

        self.env_config = env_config
        
        super().__init__(env_config, 
                         t_act,
                         num_upload_successful_videos,
                         num_upload_failed_videos,
                         show_images)
    
        
    def render_function(self, caller_locals: dict) -> np.ndarray:
        # Override this function to return an arbitrary array.
        # TODO: remove this function once we can actually render the simulation
        return np.zeros((1, 1))

    def reset_env(self, caller_locals: dict) -> np.ndarray:
        """Function to modify reset behavior in subclasses.

        For example for setting a random seed, or passing an options dict.

        """

        return self.env.reset()

    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        self.num_successful_trajectories = 0
        self.num_failed_trajectories = 0
        
        self.num_successful_lifted_hood = 0
        self.num_failed_lifted_hood = 0

        frames_of_successful_trajectories = []
        frames_of_failed_trajectories = []

        # Loop over the number of trajectories
        for i in (pbar := tqdm(range(num_trajectories), desc="Testing agent", leave=False)):
            self.reset_env(caller_locals=locals())

            done = False
            successful = False
            lifted_hood = False
            # successful_grabbed_hood = False
            episode_frames = []
            image_shape = self.render_function(caller_locals=locals()).shape

            while not done:
                observation_buffer = self.env.get_observation_dict()

                # Create torch tensors from numpy arrays
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
                assert actions.ndim == 2, f"Actions should be of shape (T, N) for T timesteps and N action dimensions. Got shape {actions.shape}"

                # Execute up to t_act actions in the environment
                for action in actions[: self.t_act]:
                    # Take action in environment
                    env_obs, reward, terminated, truncated, info = self.env.step(action)

                    # Render environment
                    #rgb_frame = self.render_function(caller_locals=locals())
                    #episode_frames.append(rgb_frame)

                    #if self.show_images:
                    #    cv2.imshow("Workspace", rgb_frame[..., ::-1])
                    #    cv2.waitKey(1)

                    successful = self.check_success_hook(caller_locals=locals())
                    if self.check_lifted_hood_hook(caller_locals=locals()): 
                        lifted_hood = True

                    # Check if episode is done
                    done = truncated or terminated

                    self.post_step_hook(caller_locals=locals())

                    # End early if episode is done
                    if done:
                        break

            # Add frames of episode to buffer of successful or failed trajectories
            if successful:
                if self.num_successful_trajectories < self.num_upload_successful_videos:
                    frames_of_successful_trajectories.extend(episode_frames)
                self.num_successful_trajectories += 1
            else:
                if self.num_failed_trajectories < self.num_upload_failed_videos:
                    frames_of_failed_trajectories.extend(episode_frames)
                self.num_failed_trajectories += 1
            
            # Update the number of trajectories where the hood was lifted or not
            if lifted_hood:
                self.num_successful_lifted_hood += 1
            else:
                self.num_failed_lifted_hood += 1


            pbar.set_postfix(success_rate=self.num_successful_trajectories / (self.num_successful_trajectories + self.num_failed_trajectories))

            self.post_episode_hook(caller_locals=locals())

        # Log at least one black frame to have info for this epoch
        fps = int(1 / self.env.dt)
        if len(frames_of_successful_trajectories) == 0:
            frames_of_successful_trajectories.append(np.zeros(image_shape, dtype=np.uint8))
        if len(frames_of_failed_trajectories) == 0:
            frames_of_failed_trajectories.append(np.zeros(image_shape, dtype=np.uint8))
        #self.log_video(frames_of_successful_trajectories, fps=fps, metric="successful")
        #self.log_video(frames_of_failed_trajectories, fps=fps, metric="failed")

        return self.get_result_dict(caller_locals=locals())


    
    def check_lifted_hood_hook(self, caller_locals: Dict) -> bool:
        return self.env.check_lifted_hood()


    def get_result_dict(self, caller_locals: Dict) -> Dict[str, float]:
        """Function to modify result dict in subclasses.

        For example adding information to the progress bar.
        """
        result_dict = {}
        result_dict["success_rate"] = self.num_successful_trajectories / (self.num_successful_trajectories + self.num_failed_trajectories)
        result_dict["success_rate_lifting_hood"] = self.num_successful_lifted_hood / (self.num_successful_lifted_hood + self.num_failed_lifted_hood)
        result_dict["reward"] = result_dict["success_rate"] + (0.1 * result_dict["success_rate_lifting_hood"])
        return result_dict