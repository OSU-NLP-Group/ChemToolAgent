import logging

from .tool_agent import ToolAgent
from .rephrasing_agent import RephrasingAgent


logger = logging.getLogger(__name__)
print_logger = logging.getLogger('chemagent_print')


class ChemAgent(object):
    def __init__(
        self,
        tools=None,
        model='gpt-4o-2024-08-06',
        tool_agent_model=None,
        tools_model=None,
        rephrasing_agent_model=None,
        api_keys={},
        max_iterations=40,
        init_tools=False,
        include_tools=None,
        exclude_tools=None,
    ):
        if tool_agent_model is None:
            tool_agent_model = model
            logger.info(f"Using model {model} for tool agent.")
        if tools_model is None:
            tools_model = model
            logger.info(f"Using model {model} for tools.")
        if rephrasing_agent_model is None:
            rephrasing_agent_model = model
            logger.info(f"Using model {model} for rephrasing agent.")
        
        """Initialize ChemAgent."""
        self.max_iterations = max_iterations

        self.tool_agent = ToolAgent(
            tools=tools,
            model=tool_agent_model,
            tools_model=tools_model,
            api_keys=api_keys,
            max_iterations=max_iterations,
            init_tools=init_tools,
            include_tools=include_tools,
            exclude_tools=exclude_tools,
        )

        self.rephrasing_agent = RephrasingAgent(
            model=rephrasing_agent_model,
            api_keys=api_keys,
        )

    def run(self, request, do_rephrasing=False, format=None, demonstration=None, verbose=True, conv_id=None):
        request = request.strip()

        tool_use_chain, conversation, conversation_with_icl = self.tool_agent.run(request, demonstration=demonstration, verbose=verbose, conv_id=conv_id)
        
        assert tool_use_chain[-1]['tool'] == 'Answer', f"Last tool in tool_use_chain is not 'Answer'. It is {tool_use_chain[-1]['tool']}."
        answer_output = tool_use_chain[-1]['output']
        direct_answer = answer_output
            
        if do_rephrasing is False:
            final_answer = direct_answer
        else:
            final_answer = self.rephrasing_agent.run(request, format, conversation=conversation)

        if verbose:
            print_logger.info('Final Answer: %s' % final_answer)

        return final_answer, tool_use_chain, conversation, conversation_with_icl
