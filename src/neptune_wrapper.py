import neptune
from dotenv import load_dotenv

load_dotenv()
import os
import cv2

NEPTUNE_API_KEY = os.getenv('NEPTUNE_API_TOKEN')
NEPTUNE_PROJECT = os.getenv('NEPTUNE_PROJECT_NAME')


class NeptuneModels:
    """
    Wrapper class for creating and modifying models in neptune.ai
    """

    def __init__(self):
        pass

    def create_model(self, model_key, model_name, model_info=None):
        """
        Init a new neptune model
        model key format ALL CAPS EG: MODEL_ONE
        model name any given string
        """
        model = neptune.init_model(
            key=model_key,
            name=model_name,
            project=NEPTUNE_PROJECT,
            api_token=NEPTUNE_API_KEY,
        )
        if (model_info != None):
            model["model"] = model_info
        model.stop()

    def model_version(self, model_key, model_params, folders_to_track = None):
        """
            Create a new version of a trained model.
            Give it the key of the existing model, the model_params used to train.
            Any optional datafiles wanted in a dict with paths to existing files e.g. {
                "replay_buffer" : "replay_buffer.csv"
            }
            Give a path to a saved binary, eg "model.pt"

        """
        model_version = neptune.init_model_version(
            model = model_key,
            project=NEPTUNE_PROJECT,
            api_token=NEPTUNE_API_KEY
        )
        model_version["model/parameters"] = model_params
        if folders_to_track != None:
            for i in range(len(folders_to_track)):
                print(folders_to_track[i])
                model_version["model/dataset"].upload_files(folders_to_track[i])
        model_version.wait()
        model_version.stop()


class NeptuneRun:
    """
    Wrapper class for interacting with neptune.ai API. Should be reinitialized
    """

    def __init__(self, params, description = "", tags = []):
        self.run = neptune.init_run(
            project=NEPTUNE_PROJECT,
            api_token=NEPTUNE_API_KEY,
            description = description,
            tags = tags
        )  # your credentials
        self.params = params
        self.run["parameters"] = self.params

    def log_run(self, metadata_dict):
        """
         Logs a given metadata for an entire run containing e.g. avg reward, avg loss etc.
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
        if metadata_dict != None:
            for key in (metadata_dict):
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
        if metadata_dict != None:
            for key in (metadata_dict):
                self.run[key].append(metadata_dict[key])

    def log_lists(self, metadata_dict):
        if metadata_dict != None:
            for key in (metadata_dict):
                self.run[key].extend([metadata_dict[key] for metadata_dict[key] in range(len(metadata_dict[key]))])

    def log_frames(self, frames):
        x, y, z = frames[0].shape
        size = (y, x)
        vid = cv2.VideoWriter('animation.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, size, True)
        for frame in frames:
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vid.write(rgb_img)
        vid.release()
        self.run["train/animation"].upload('animation.mp4')
        self.run.wait()
        os.remove('animation.mp4')

    def finish(self):
        self.run.stop()

    def __del__(self):
        if (self.run.get_state != 'stopped'):
            self.run.stop()

