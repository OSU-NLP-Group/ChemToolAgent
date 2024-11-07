from .base import BaseTool
from chemagent.llms import make_llm


SYSTEM_PROMPT = 'You are an expert chemist. Your task is to answer the following question to your best ability. You can think step by step, and your answer should contain all the information necessary to answer the question.'


class AiExpert(BaseTool):
    name = "AiExpert"
    func_name = "ask_ai_expert"
    description = "An AI expert that can answer any questions. When there are no other tools that can meet your need, this tool can be used as the last resort. Input your question, returns the answer."
    func_doc = ("question: str", "str")
    func_description = "Ask the AI expert any question. When there are no other tools that can meet your need, this tool can be used as the last resort. Input your question, returns the answer."
    examples = [
        {'input': 'What is the boiling point of water?', 'output': 'The boiling point of water at standard atmospheric pressure (1 atmosphere or 101.325 kPa) is 100°C (212°F). However, the boiling point can vary depending on the surrounding atmospheric pressure. For example, at higher altitudes where the atmospheric pressure is lower, water boils at a temperature lower than 100°C.'},
    ]

    def __init__(self, api_keys, model="gpt-4-turbo-2024-04-09", init=True, interface='text'):
        super().__init__(init, interface=interface)
        self.llm = make_llm(model, api_keys)

    def _run_base(self, query: str, *args, **kwargs) -> str:
        conv = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': 'Question: ' + query}
        ]
        r = self.llm.request(conv)[0]
        return r
