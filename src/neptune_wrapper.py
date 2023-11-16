import os

import cv2
import neptune
from dotenv import load_dotenv

load_dotenv()

NEPTUNE_API_KEY = os.getenv("NEPTUNE_API_TOKEN")
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT_NAME")


class NeptuneModels:
    """
    Wrapper class for creating and managing models in neptune.ai.
    This class provides functionality to create new models and model versions
    in Neptune's ML metadata store.
    """

    def __init__(self):
        pass

    def create_model(self, model_key, model_name, model_info=None):
        """
        Create a new model entry in Neptune.

        Parameters:
        model_key (str): The unique identifier for the model in ALL CAPS.
        model_name (str): A human-readable name for the model.
        model_info (dict, optional): Additional information about the model.

        The method initializes a new model in Neptune
        with the given key, name, and optional
        info, then stops the model's logger after creation.
        """
        model = neptune.init_model(
            key=model_key,
            name=model_name,
            project=NEPTUNE_PROJECT,
            api_token=NEPTUNE_API_KEY,
        )
        if model_info is not None:
            model["model"] = model_info
        model.stop()

    def model_version(self, model_key, model_params, folders_to_track=None):
        """
        Create a new version of a trained model.
        Give it the key of the existing model, the model_params used to train.
        Any optional datafiles wanted in a dict with paths to existing files e.g. {
            "replay_buffer" : "replay_buffer.csv"
        }
        Give a path to a saved binary, eg "model.pt"
        """
        model_version = neptune.init_model_version(
            model=model_key, project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_KEY
        )
        model_version["model/parameters"] = model_params
        if folders_to_track is not None:
            for i in range(len(folders_to_track)):
                print(folders_to_track[i])
                model_version["model/dataset"].upload_files(folders_to_track[i])
        model_version.wait()
        model_version.stop()


class NeptuneRun:
    """
    Wrapper class for interacting with neptune.ai API. Should be reinitialized
    """

    def __init__(self, params, description="", tags=[]):
        """
        Initializes a Neptune run with provided parameters, description, and tags.

        Parameters:
        params (dict): Parameters for the run to log.
        description (str): A short description of the run.
        tags (list of str): A list of tags for categorizing the run.

        The run is initialized with the given project, API token, description, and tags.
        It logs the parameters immediately upon starting.
        """
        self.run = neptune.init_run(
            project=NEPTUNE_PROJECT,
            api_token=NEPTUNE_API_KEY,
            description=description,
            tags=tags,
        )  # your credentials
        self.params = params
        self.run["parameters"] = self.params

    def log_run(self, metadata_dict):
        """
         Logs a given metadata for an entire run
         containing e.g. avg reward, avg loss etc.
        Call for a run if metadata is relevant to entire run.
        format:
        {
            "loss" : 1
            "reward" : 1
            For folder structure
            "train/loss" : 1
            "train/reward" : 1
        }
        """
        if metadata_dict is not None:
            for key in metadata_dict:
                self.run[key].append(metadata_dict[key])

    def log_epoch(self, metadata_dict):
        """
         Logs a given metadata for an epoch containing e.g. reward, loss etc.
        Call for a run if metadata is relevant to entire run.
        format:
        {
            "loss" : 1
            "reward" : 1
            For folder structure
            "train/loss" : 1
            "train/reward" : 1
        }
        """
        if metadata_dict is not None:
            for key in metadata_dict:
                self.run[key].append(metadata_dict[key])

    def log_lists(self, metadata_dict):
        """
        Logs lists of data as metadata.

        Parameters:
        metadata_dict (dict): A dictionary where each key
        corresponds to a list of data points to log.

        The method extends the logged data for each key
        with the corresponding list of values, logging each
        index as a separate entry.
        """
        if metadata_dict is not None:
            for key in metadata_dict:
                self.run[key].extend(metadata_dict[key])

    def log_frames(self, frames, episode_number):
        """
        Creates a video from a list of frames representing one episode
        of the game and uploads it to the logging system.
        Each video is named with the episode number to
        differentiate between different episodes. After uploading, the
        local video file is deleted to save space.

        Parameters:
        frames (list): A list of frames to be combined into a video.
        episode_number (int): The episode number to be included in the video filename.

        """
        x, y, z = frames[0].shape
        size = (y, x)
        vid = cv2.VideoWriter(
            "animation_{}.mp4".format(episode_number),
            cv2.VideoWriter_fourcc(*"XVID"),
            30,
            size,
            True,
        )
        for frame in frames:
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vid.write(rgb_img)
        vid.release()
        self.run["train/animation"].upload_files(
            "animation_{}.mp4".format(episode_number)
        )
        self.run.wait()
        os.remove("animation_{}.mp4".format(episode_number))

    def finish(self):
        """
        Stops the neptune run, finalizing the logging process.
        This method should be called when all logging for the run
        is complete and the run is ready to be closed.
        """
        self.run.stop()

    def __del__(self):
        """
        Destructor for the NeptuneRun class.
        Ensures that the Neptune run is stopped and resources are
        cleaned up when the NeptuneRun object is destroyed.
        This is a safeguard to stop the run in case it was not stopped manually.
        """
        if self.run.get_state != "stopped":
            self.run.stop()
