import neptune
from dotenv import load_dotenv
load_dotenv()
import os

NEPTUNE_API_KEY = os.getenv('NEPTUNE_API_TOKEN')
NEPTUNE_PROJECT = os.getenv('NEPTUNE_PROJECT_NAME')

class NeptuneRun:
    """
    Wrapper class for interacting with neptune.ai API. Should be reinitialized 
    """
    def __init__(self, params):
        self.run = neptune.init_run(
            project=NEPTUNE_PROJECT,
            api_token=NEPTUNE_API_KEY,
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
                self.run[key].extend([metadata_dict[key]  for metadata_dict[key] in range(len(metadata_dict[key]))])
    def finish(self):
        self.run.stop()
    def __del__(self):
        self.run.stop()
    

