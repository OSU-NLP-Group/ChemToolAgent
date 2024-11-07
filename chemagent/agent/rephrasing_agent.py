import logging

from chemagent.llms import make_llm


print_logger = logging.getLogger('chemagent_print')

logger = logging.getLogger(__name__)


REPHRASE_TEMPLATE = """For this task, you'll act as a scientific assistant. I'll give you a question along with a draft solution. The draft solution contains all the accurate information needed to answer the question. Your job is to write a complete, final answer to the question by using and concluding the information from the draft solution. Make sure your response includes all necessary information and reasoning to fully address the original question.

Question: {question}

Solution draft:
{agent_ans}

{format_requirement}"""



class RephrasingAgent(object):
    def __init__(
        self,
        model="gpt-4-0613",
        api_keys={},
    ):
        self.llm = make_llm(model, api_keys)

    def run(self, request, format=None, conversation=None, draft=None, verbose=True):
        if conversation is not None:
            draft = self._construct_draft(conversation)
        else:
            assert draft is not None
            if verbose:
                logger.debug('Using direct draft for rephrasing.')

        rephrase_prompt = REPHRASE_TEMPLATE.format(
            question=request, agent_ans=draft, format_requirement='' if format is None else ('Format requirement: ' + format)
        )
        rephrase_conversion = [
            {'role': 'user', 'content': rephrase_prompt}
        ]
        prefix = 'Certainly. Here\'s the final answer to the question based on the draft solution:'
        final_answer = self.llm.request(rephrase_conversion, prefix=prefix)[0][len(prefix):].strip()

        return final_answer

    def _construct_draft(self, conversation):
        draft = "===== Draft Start =====\n"
        conversation = conversation[2:]
        idx = 1
        for item in conversation:
            role = item['role']
            if role == 'assistant':
                draft += "--- Step {idx} ---\n{content}\n".format(idx=idx, content=item['content'])
                idx += 1
            else:
                draft += item['content'] + '\n\n'
        
        draft += "\n===== Draft End ====="
        
        return draft
