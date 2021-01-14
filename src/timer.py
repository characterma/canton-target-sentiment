import time
import json


class Timer:
    def __init__(self, output_dir):
        self.preprocessing_start_time = None
        self.inference_start_time = None
        self.durations = {
            "preprocessing": 0, 
            "inference": 0,  
        }
        self.output_dir = output_dir

    def on_preprocessing_start(self):
        self.preprocessing_start_time = time.time()

    def on_preprocessing_end(self):
        self.durations["preprocessing"] += time.time() - self.preprocessing_start_time

    def on_inference_start(self):
        self.inference_start_time = time.time()

    def on_inference_end(self):
        self.durations["inference"] += time.time() - self.inference_start_time

    # @property
    # def durations(self):
    #     return self.durations

    def save_timer(self):
        with open(self.output_dir / "timer.json", 'w') as fp:
            json.dump(self.durations, fp)

