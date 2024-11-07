import re
import logging
import requests
import json
import time

from uuid import uuid4

from chemagent.utils.error import *
from .base import BaseTool

logger = logging.getLogger(__name__)


def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """
    query = query.strip()
    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    query = query.lstrip('"').lstrip("'").lstrip()
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    query = query.rstrip('"').rstrip("'").rstrip()
    return query


class ClientJupyterKernel:
    def __init__(self, url):
        self.url = url

    def execute(self, code, conv_id):
        payload = {"convid": conv_id, "code": code}
        response = requests.post(self.url, data=json.dumps(payload))
        response_data = response.json()
        try:
            if response_data["new_kernel_created"]:
                print(f"New kernel created for conversation {conv_id}")
        except KeyError:
            print(response_data)
            raise
        return response_data["result"]


class PythonShell(BaseTool):
    name: str = "PythonREPL"
    func_name: str = 'run_python_code'
    description: str = "A Python shell that can execute python commands. Input should be a valid python command. You can input `!pip install ...` to install packages if needed."
    func_doc = ("code: str", "str")
    func_description = description
    examples = [
        {'input': 'print("Hello, World!")', 'output': 'Hello, World!'},
    ]

    def __init__(self, input_sanitize=True, init=True, interface='text', url='http://localhost:8888/execute', max_pool=5) -> None:
        super().__init__(init, interface)
        self.url = url
        self.input_sanitize = input_sanitize
        self.client = ClientJupyterKernel(self.url)

    def _run_base(self, query: str, conv_id: str = 'default', *args, **kwargs):
        try:
            if self.input_sanitize:
                query = sanitize_input(query)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            raise ChemAgentInputError(f"An error occurred while sanitizing the input python code: {str(e)}")
        
        k = 0
        while True:
            try:
                r = self.client.execute(query, conv_id)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                raise ChemAgentFatalError(f"An error occurred while running the python code: {str(e)}")
            
            if r == '[Code executed successfully with no output]' and 'print' in query:
                logger.warning('No output:\n\n' + query)
                conv_id = str(uuid4())
                k += 1
                if k > 3:
                    break
                else:
                    time.sleep(3)
                    logger.info('retrying')
                    continue
            
            break

        return r


if __name__ == "__main__":
    # Example usage
    python_shell = PythonShell()
    print(python_shell.run_text("print('Hello, World!')", 'test'))
    