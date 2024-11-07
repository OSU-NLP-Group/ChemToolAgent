import openai
import time
import json
from copy import deepcopy

from .requester import LLMRequester

class GptRequester(LLMRequester):
    def __init__(self, api_code, model_name='gpt-4', trial_time=1, sleep_time=5):
        super().__init__(api_code, model_name, trial_time, sleep_time)
        openai.api_key = self.api_code

    def request(self, conversation, num_return=1, prefix=None, stop_sequences=None):
        if stop_sequences is not None:
            raise NotImplementedError("stop_sequences is not supported for GPT.")
        conversation = deepcopy(conversation)
        if prefix is not None:
            assert conversation[-1]['role'] == 'user'
            prefix = prefix.rstrip()
            conversation.append({'role': 'assistant', 'content': prefix})

        k = 0
        while True:
            try:
                r = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=conversation,
                    n=num_return,
                )
            except openai.error.Timeout:
                if k >= self.trial_time:
                    raise
                k += 1
                time.sleep(self.sleep_time)
                continue
            else:
                break
        
        output_list = []
        for item in r['choices']:
            response = item['message']['content'].rstrip()
            output_list.append((prefix + response) if prefix is not None else response)
        
        return output_list
    
class NewGptRequester(LLMRequester):
    def __init__(self, api_code, model_name='gpt-4', trial_time=1, sleep_time=5, use_user_prompt_for_system_prompt=False):
        super().__init__(api_code, model_name, trial_time, sleep_time)
        self.client = openai.OpenAI(api_key=self.api_code)
        self._batched_request = []
        self.custom_ids = set()
        self.use_user_prompt_for_system_prompt = use_user_prompt_for_system_prompt

    def request(self, conversation, num_return=1, prefix=None, stop_sequences=None):
        conversation = deepcopy(conversation)
        if prefix is not None:
            assert conversation[-1]['role'] == 'user'
            prefix = prefix.rstrip()
            conversation.append({'role': 'assistant', 'content': prefix})

        system_prompt = None
        if conversation[0]['role'] == 'system':
            system_prompt = conversation[0]['content']
        
        if system_prompt is not None and self.use_user_prompt_for_system_prompt:
            conversation = conversation[1:]
            assert conversation[0]['role'] == 'user'
            conversation[0]['content'] = system_prompt + '\n\n' + conversation[0]['content']

        k = 0
        while True:
            try:
                if stop_sequences is None:
                    r = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=conversation,
                        n=num_return,
                    )
                else:
                    r = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=conversation,
                        n=num_return,
                        stop=stop_sequences,
                    )
            except openai.APITimeoutError:
                if k >= self.trial_time:
                    raise
                k += 1
                time.sleep(self.sleep_time)
                continue
            else:
                break
        
        # TODO: Add log when model stopped due to stop_sequences
        output_list = []
        for item in r.choices:
            response = item.message.content.rstrip()
            output_list.append((prefix + response) if prefix is not None else response)
        
        return output_list
    
    def add_request(self, conversation, custom_id, num_return=1, *args, **kwargs):
        if num_return != 1:
            raise NotImplementedError("num_return must be 1 for batch_request. Whether more than 1 is supported is not checked yet.")
        if custom_id in self.custom_ids:
            raise ValueError("custom_id already exists.")
        self._batched_request.append((conversation, custom_id, args, kwargs))

    def clear_request(self):
        self._batched_request.clear()

    def get_batched_request(self):
        requests = []
        conversations = {}
        other_info = {}
        for idx, (conv, custom_id, args, kwargs) in enumerate(self._batched_request):
            if custom_id is None:
                custom_id = str(idx)
            request_item = {
                'custom_id': custom_id,
                'method': "POST",
                'url': "/v1/chat/completions",
                'body': {
                    'model': self.model_name,
                    'messages': conv,
                    # 'n': num_return,
                }
            }
            requests.append(request_item)
            conversations[custom_id] = conv
            other_info[custom_id] = (args, kwargs)
        return requests, conversations, other_info
    
    def batch_request(self):
        if self._batched_request is None or len(self._batched_request) == 0:
            return {}
        requests, conversations, other_info = self.get_batched_request()
        num_requests = len(requests)
        requests_jsonl_str = '\n'.join([json.dumps(item) for item in requests])
        requests_jsonl_str_bytes = requests_jsonl_str.encode('utf-8')
        batch_input_file = self.client.files.create(
            file=requests_jsonl_str_bytes,
            purpose='batch'
        )
        tmp = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint='/v1/chat/completions',
            completion_window='24h',
            metadata={
                'description': 'Batch request for chat completions.',
            }
        )
        batch_id = tmp.id
        print('Batch ID:', batch_id)
        print('Running batch request for %d requests ...' % num_requests)
        time.sleep(1 * num_requests)
        try:
            while True:
                batch_object = self.client.batches.retrieve(batch_id)
                if batch_object.status == 'completed':
                    break
                elif batch_object.status == 'failed':
                    raise RuntimeError("Batch request failed: %s" % batch_object.errors.message)
                time.sleep(5)
        except KeyboardInterrupt:
            print('Current batch_output:', batch_object)
            raise
        output_file_id = batch_object.output_file_id
        content = self.client.files.content(output_file_id)
        lines = content.content.decode('utf-8').split('\n')
        outputs = {}
        failed_samples = []
        for line in lines:
            if line.strip() == '':
                continue
            item = json.loads(line)
            
            try:
                output = []
                for choice in item['response']['body']['choices']:
                    response = choice['message']['content'].strip()
                    output.append(response)
                outputs[item['custom_id']] = (conversations[item['custom_id']], output, other_info[item['custom_id']])
            except KeyError:
                failed_samples.append(item['custom_id'])
            
        return outputs, failed_samples
