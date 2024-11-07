import molbloom
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from tavily import TavilyClient

from ..utils.error import *
from ..tools import BaseTool
from ..utils import is_smiles
    

class WebSearch(BaseTool):
    name = "WebSearch"
    func_name = 'search_web'
    description = "Search the web for any questions and knowledge (including both general ones and domain-specific ones) and obtain concise summaries of the most relevant content. Input a specific question, returns a summary of the relevant content that answers the question."
    func_doc = ("question: str", "str")
    func_description = description
    examples = [
        {'input': 'What is the boiling point of water?', 'output': 'The boiling point of water at sea level is 100°C (212°F).'},
    ]

    def __init__(self, tavily_api_key: str, init=True, interface='text'):
        assert tavily_api_key is not None
        self.client = TavilyClient(api_key=tavily_api_key)
        super().__init__(init, interface=interface)

    def _run_base(self, query: str, *args, **kwargs) -> str:
        response = self.client.search(query, search_depth='advanced', include_answer=True)
        answer = response['answer']
        return answer


class PatentCheck(BaseTool):
    name = "PatentCheck"
    func_name = 'check_if_patented'
    description = "Input SMILES of a molecule (one at a time), returns if molecule is patented."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'CCO', 'output': 'not patented'},
    ]

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        """Checks if compound is patented. Give this tool only one SMILES string"""
        if not is_smiles(smiles):
            raise ChemAgentInputError('The input is not a valid SMILES representation. Please double-check and make sure that you only input one molecule.')
        try:
            r = molbloom.buy(smiles, canonicalize=True, catalog="surechembl")
            if r:
                output = "patented"
            else:
                output = "not patented"
        except KeyboardInterrupt:
            raise
        except:
            raise ChemAgentInputError('The input is not a valid SMILES representation. Please double-check and make sure that you only input one molecule.')
        return output


class Wikipedia(BaseTool):
    """Tool that searches the Wikipedia API."""

    name: str = "WikipediaSearch"
    func_name: str = 'search_wikipedia'
    description: str = "Search Wikipedia. Input a search query, returns summaries of related content."
    func_doc = ("query: str", "str")
    func_description = description
    examples = [
        {'input': 'Water', 'output': 'Page: Water\nSummary: Water is an inorganic compound with the chemical formula H2O. It is a transparent, tasteless, odorless, and nearly colorless chemical substance, and it is the main constituent of Earth\'s hydrosphere and the fluids of all known living organisms (in which it acts as a solvent). [...]'},
    ]

    def __init__(self, init=True, interface='text') -> None:
        self.api_wrapper = WikipediaAPIWrapper()
        super().__init__(init, interface)

    def _run_base(self, query: str, *args, **kwargs) -> str:
        """Use the Wikipedia tool."""
        return self.api_wrapper.run(query)

