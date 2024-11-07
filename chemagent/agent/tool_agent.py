import logging
import sys

from chemagent.utils.error import *
from chemagent.llms import make_llm
from chemagent.agent.tools import make_tools, verify_tools, PythonShell, AiExpert


print_logger = logging.getLogger('chemagent_print')

logger = logging.getLogger(__name__)

THOUGHT_TITLE = 'Thought'
ACTION_TITLE = 'Tool'
ACTION_INPUT_TITLE = 'Tool Input'
OBSERVATION_TITLE = 'Tool Output'
ANSWER_TITLE = 'Answer'

THOUGHT_TITLE_SC = THOUGHT_TITLE + ':'
ACTION_TITLE_SC = ACTION_TITLE + ':'
ACTION_INPUT_TITLE_SC = ACTION_INPUT_TITLE + ':'
OBSERVATION_TITLE_SC = OBSERVATION_TITLE + ':'
ANSWER_TITLE_SC = ANSWER_TITLE + ':'


PREFIX = """You are an expert chemist. Your task is to use the provided tools and respond to the input question to the best of your ability.

"""

FORMAT_INSTRUCTIONS = """You must respond in one of two specific formats in every step:

1. When calling a tool:
    {thought_title}: [Your reasoning for the next step]
    {action_title}: [Exact name of the tool to use, must be one from the provided list: {tool_names}]
    {action_input_title}: [Specific input for the selected tool. You must add "<END_INPUT>" at the end of the input to indicate the end position.]
   Then you will be provided with the tool output.
2. When providing the final answer after obtaining all necessary information with tools:
    {thought_title}: [Conclusion of the gathered information and reasoning for the final answer]
    {answer_title}: [Your conclusion based on gathered information and comprehensive response to the original question]

Guidelines:
- You should call tools to solve the problem, especially when you are not sure about certain things and when tools can help.
- If no other tools are suitable, use the {llm_tool_name} tool to ask questions and obtain analysis.
- Only provide the final answer after you have gathered all necessary information using tools.
- Always use the exact format specified, including the colons after "{thought_title}", "{action_title}", "{action_input_title}", and "{answer_title}". Do not use any other format or include any additional text outside these structures.
- You can only call one tool at a time. Once you have output "{action_title}" and "{action_input_title}" for a tool, please stop generating text and wait for the tool output.


The provided tools:

{tool_strings}


Use the above tools to respond to the user's question.
""".format(thought_title=THOUGHT_TITLE, action_title=ACTION_TITLE, action_input_title=ACTION_INPUT_TITLE, answer_title=ANSWER_TITLE, llm_tool_name=AiExpert.name, tool_names='{{{tool_names}}}', tool_strings='{tool_strings}')


QUESTION_PROMPT = """Question: {input}

"""


def extract_tool_command(text):
    thought_pos = text.find(THOUGHT_TITLE_SC)
    action_pos = text.find(ACTION_TITLE_SC)
    assert action_pos != -1, text
    if thought_pos == -1:
        thought = None
    else:
        assert thought_pos < action_pos
        if text[:thought_pos].strip() == '':
            thought = text[thought_pos + len(THOUGHT_TITLE_SC): action_pos].strip()
        else:
            thought = text[:action_pos].replace(THOUGHT_TITLE_SC, '').strip()
    input_pos = text.find(ACTION_INPUT_TITLE_SC)
    assert action_pos < input_pos
    action = text[action_pos + len(ACTION_TITLE_SC): input_pos].strip()
    action_input = text[input_pos + len(ACTION_INPUT_TITLE_SC):].strip()
    return thought, action, action_input


def extract_command(text):
    if ACTION_TITLE_SC in text:
        if ACTION_INPUT_TITLE_SC not in text:
            raise ChemAgentOutputError("The output contains \"%s\" but does not contain \"%s\": %s" % (ACTION_TITLE_SC, ACTION_INPUT_TITLE_SC, text))
        num_action = text.count(ACTION_TITLE_SC)
        num_action_input = text.count(ACTION_INPUT_TITLE_SC)
        if num_action > 1 or num_action_input > 1:
            raise ChemAgentOutputError("The output contains more than one \"%s\" or \"%s\": %s" % (ACTION_TITLE_SC, ACTION_INPUT_TITLE_SC, text))
        if num_action != num_action_input:
            raise ChemAgentOutputError("The output contains different number of \"%s\" and \"%s\": %s" % (ACTION_TITLE_SC, ACTION_INPUT_TITLE_SC, text))
        num_action_input = text.count(ACTION_INPUT_TITLE_SC)
        return extract_tool_command(text)
    elif ACTION_INPUT_TITLE_SC in text:
        raise ChemAgentOutputError("The output contains \"%s\" but does not contain \"%s\": %s" % (ACTION_INPUT_TITLE_SC, ACTION_TITLE_SC, text))
    else:
        if ANSWER_TITLE_SC in text:
            final_answer_pos = text.rfind(ANSWER_TITLE_SC)
            final_answer = text[final_answer_pos + len(ANSWER_TITLE_SC):].strip()
            if text[: final_answer_pos].strip() == '':
                thought = None
            else:
                if THOUGHT_TITLE_SC in text:
                    thought_pos = text.rfind(THOUGHT_TITLE_SC)
                    assert thought_pos < final_answer_pos
                    if text[:thought_pos].strip() == '':
                        thought = text[thought_pos + len(THOUGHT_TITLE_SC): final_answer_pos].strip()
                    else:
                        thought = text[:final_answer_pos].strip()
                else:
                    thought = text[:final_answer_pos].strip()
        else:
            thought = None
            final_answer = text
            logger.info("The output does not contain \"%s\", but regarded as the final answer anyway." % (ANSWER_TITLE_SC))
        return thought, None, final_answer


def construct_tool_example_string(tool):
    examples = tool.__class__.examples
    example_strings = []
    for example in examples:
        example_string = 'Tool Input: {input}\nTool Output: {output}'.format(input=example['input'], output=example['output'])
        example_strings.append(example_string)
    return '\n'.join(example_strings)


class ToolAgent(object):
    def __init__(
        self,
        tools=None,
        model="gpt-4-0613",
        tools_model="gpt-3.5-turbo-0613",
        api_keys={},
        max_iterations=40,
        max_error_iterations=3,
        init_tools=True,
        include_tools=None,
        exclude_tools=None,
    ):
        """Initialize ChemAgent."""
        self.max_iterations = max_iterations
        self.max_error_iterations = max_error_iterations

        self.llm = make_llm(model, api_keys)
        
        if tools is None:
            tools = make_tools(tools_model, api_keys=api_keys, init=init_tools, include_tools=include_tools, exclude_tools=exclude_tools)
        
        missing_tools, extra_tools, duplicate_tools = verify_tools(tools)
        abnormal = False
        if len(missing_tools) > 0:
            logger.info('Missing tools: ' + ', '.join(missing_tools))
            abnormal = True
        if len(extra_tools) > 0:
            logger.info('Extra tools: ' + ', '.join(extra_tools))
            abnormal = True
        if len(duplicate_tools) > 0:
            logger.info('Duplicate tools: ' + ', '.join(duplicate_tools))
            abnormal = True
        if abnormal:
            logger.info('Equipped tools: ' + ', '.join([tool.name for tool in tools]))
            c = input('Abnormal tools. Continue? (y/n): ')
            if c.lower() != 'y':
                sys.exit(0)

        self.tools = tools
        self.tool_dict = {}
        for tool in self.tools:
            name = tool.name
            self.tool_dict[name] = tool

        self.format_instructions = FORMAT_INSTRUCTIONS
        if AiExpert.name not in self.tool_dict or (AiExpert.name in self.tool_dict and len(self.tools) == 1):
            self.format_instructions = self.format_instructions.replace(
                '- If no other tools are suitable, use the {llm_tool_name} tool to ask questions and obtain analysis.'.format(llm_tool_name=AiExpert.name) + '\n',
                ''
            )

        self.tool_names = tuple([tool.name for tool in self.tools])
        self.tool_strings = '\n\n'.join(
            [
                '{name}: {description}'.format(name=tool.name, description=tool.description) # construct_tool_example_string(tool))
                for tool in self.tools
            ]
        )

    def run(self, request, demonstration=None, verbose=True, conv_id=None):
        tool_names = ', '.join(self.tool_names)
        tool_strings = self.tool_strings

        tool_use_chain = []

        conversation = [
            {'role': 'system', 'content': PREFIX + self.format_instructions.format(tool_names=tool_names, tool_strings=tool_strings)},
        ]

        if demonstration is not None:
            len_demonstration = len(demonstration)
            assert len_demonstration >= 2

            for item in demonstration:
                if item['role'] == 'assistant' and 'Tool Input' in item['content']:
                    item['content'] = item['content'].rstrip()+'\n<END_INPUT>'

            conversation = conversation + demonstration
        else:
            len_demonstration = 0

        conversation.append(
            {
                'role': 'user', 
                'content': QUESTION_PROMPT.format(input=request),
            }
        )

        enable_print = True
        idx = 1
        error_iterations = 0
        while True:
            if idx > self.max_iterations:
                raise RuntimeError("Running exceeds the max iteration limit (%d)." % self.max_iterations)

            llm_output = self.llm.request(conversation, prefix=None, stop_sequences=['<END_INPUT>'])[0]
            if verbose and enable_print:
                print_logger.info('--- Step %d ---' % idx)
                
            try:
                thought, action, action_input = self._extract_command(llm_output)
            except (ChemAgentOutputError, AssertionError):
                enable_print = False
                error_iterations += 1
                if error_iterations >= self.max_error_iterations:
                    raise ChemAgentOutputError("Failed to extract command from the output after %d iterations.\n%sn" % (self.max_error_iterations, llm_output))
                logger.debug('Failed to extract command from:\n' + llm_output + '\n\n')
                continue
            else:
                enable_print = True

            if action is None:
                new_line = {
                    'role': 'assistant', 
                    'content': llm_output,
                }
                conversation.append(new_line)
                tool_use_chain.append(
                    {'thought': thought, 'tool': 'Answer', 'input': None, 'output': action_input, 'success': True, 'raw_output': llm_output}
                )

                if verbose:
                    print_logger.info(llm_output + '\n\n')

                break
            else:
                new_line = {
                    'role': 'assistant',
                    'content': llm_output.rstrip() + '\n<END_INPUT>' if ACTION_INPUT_TITLE_SC in llm_output else llm_output,
                }
                conversation.append(new_line)

                if verbose:
                    print_logger.info(llm_output)
                
                success, tool_result = self._call_tool(action, action_input, conv_id=conv_id)
                new_line = {
                    'role': 'user',
                    'content': '%s ' % OBSERVATION_TITLE_SC + str(tool_result),
                }
                conversation.append(new_line)
                tool_use_chain.append(
                    {'thought': thought, 'tool': action, 'input': action_input, 'output': str(tool_result), 'success': success, 'raw_output': llm_output}
                )

                if verbose:
                    print_logger.info('%s %s\n\n' % (OBSERVATION_TITLE_SC, str(tool_result)))

                idx += 1
        
        original_conversation = conversation
        if demonstration is not None:
            conversation = conversation[:1] + conversation[1 + len_demonstration:]

        return tool_use_chain, conversation, original_conversation

    def _extract_command(self, text):
        return extract_command(text)

    def _call_tool(self, tool_name, tool_input, conv_id=None):
        if tool_name not in self.tool_names:
            return False, "\"{tool_name}\" is not a valid tool. Please select tool to use from {tool_names}).".format(tool_name=tool_name, tool_names='{ ' + ', '.join(self.tool_names) + ' }')
        try:
            if tool_name == PythonShell.name and conv_id is not None:
                r = self.tool_dict[tool_name](tool_input, conv_id=conv_id)
            else:
                r = self.tool_dict[tool_name](tool_input)
            if tool_name == PythonShell.name:
                r_strip = r.strip()
                if r_strip == '[Code executed successfully with no output]':
                    raise ChemAgentFatalError('Python code executed successfully with no output. Need a check. Tool Input: \n=== Code Start ===\n%s\n=== Code End ===' % tool_input)
                elif r_strip.startswith('<Figure size') and len(r_strip) > 200:
                    raise ChemAgentOutputError('[Figure not shown]')
                
        except ChemAgentGeneralError as e:
            logger.debug("Tool that raised error: " + tool_name)
            return False, 'Error: ' + str(e)
        return True, r
