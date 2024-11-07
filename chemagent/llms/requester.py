from abc import ABC, abstractmethod

class LLMRequester(ABC):
    def __init__(self, api_code, model_name, trial_time=1, sleep_time=5):
        self.model_name = model_name
        self.api_code = api_code
        self.trial_time = int(trial_time)
        self.sleep_time = int(sleep_time)

    @abstractmethod
    def request(self, conversation, num_return=1, prefix=None):
        pass
