from anthropic import Anthropic
import time
import warnings
from copy import deepcopy
import logging

from .requester import LLMRequester


logger = logging.getLogger(__name__)


class ClaudeRequester(LLMRequester):
    def __init__(self, api_code, model_name='claude-3-opus-20240229', trial_time=1, sleep_time=5, use_user_prompt_for_system_prompt=False):
        super().__init__(api_code, model_name, trial_time, sleep_time)
        self.client = Anthropic(api_key=self.api_code)
        self.use_user_prompt_for_system_prompt = use_user_prompt_for_system_prompt

    def request(self, conversation, num_return=1, max_tokens=2048, prefix=None, stop_sequences=None):
        conversation = deepcopy(conversation)
        if prefix is not None:
            assert conversation[-1]['role'] == 'user'
            prefix = prefix.rstrip()
            conversation.append({'role': 'assistant', 'content': prefix})

        if num_return > 1:
            warnings.warn(
                "Claude API does not support num_return > 1. Using num_return = 1."
            )

        system_prompt = None
        if conversation[0]['role'] == 'system':
            system_prompt = conversation[0]['content']
            conversation = conversation[1:]
        
        if system_prompt is not None and self.use_user_prompt_for_system_prompt:
            assert conversation[0]['role'] == 'user'
            conversation[0]['content'] = system_prompt + '\n\n' + conversation[0]['content']

        k = 0
        while True:
            try:
                if system_prompt is not None:
                    if self.use_user_prompt_for_system_prompt:
                        r = self.client.messages.create(
                            max_tokens=max_tokens,
                            messages=conversation,
                            model=self.model_name,
                            stop_sequences=stop_sequences,
                        )
                    else:
                        r = self.client.messages.create(
                            max_tokens=max_tokens,
                            system=system_prompt,
                            messages=conversation,
                            model=self.model_name,
                            stop_sequences=stop_sequences,
                        )
                else:
                    r = self.client.messages.create(
                        max_tokens=max_tokens,
                        messages=conversation,
                        model=self.model_name,
                        stop_sequences=stop_sequences,
                    )
            except KeyboardInterrupt:
                raise
            except:
                if k >= self.trial_time:
                    raise
                k += 1
                time.sleep(self.sleep_time)
                continue
            else:
                if r.stop_reason == 'stop_sequence':
                    logger.info('Stop sequence detected.')
                break

        output_list = []
        response = r.content[0].text.rstrip()
        output_list.append((prefix + response) if prefix is not None else response)
        
        return output_list
