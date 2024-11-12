# ChemAgent
This code repo contains ChemAgent implementation for the paper "Tooling or Not Tooling? The Impact of Tools on Language Agents for Chemistry Problem Solving".

## Installation

**API Keys**

To use the backbone LLMs (GPT-4o and Claude-3.5-Sonnet) and some tools, you need to obtain the API keys first. Please follow the links in `api_keys.py` to get your keys and insert your keys there.

**Environment Setup**

```bash
conda create -n chemagent python=3.9
pip install -r requirements.txt

# Manually install Uni-Core, following https://github.com/dptech-corp/Uni-Core
```

## Usage

`cd` to the `python_server` folder, run:

```bash
./start_jupyter_server.sh 8888
```

Then `cd` back to the project root folder, and use the following commands to query ChemAgent:

```python
from api_keys import api_keys
from chemagent import ChemAgent

agent = ChemAgent(model='gpt-4o-2024-08-06', api_keys=api_keys)

query = "What is the molecular weight the chemical compound Caffeine."
final_answer, tool_use_chain, conversation, conversation_with_icl = agent.run(query)
```

You could play the agent in the Jupyter notebook `playground.ipynb`.

## Citation

If our paper or related resources prove valuable to your research, we kindly ask for citation. Please feel free to contact us with any inquiries.

```
@article{yu2024chemagent,
    title={Tooling or Not Tooling? The Impact of Tools on Language Agents for Chemistry Problem Solving},
    author={Botao Yu and Frazier N. Baker and Ziru Chen and Garrett Herb and Boyu Gou and Daniel Adu-Ampratwum and Xia Ning and Huan Sun},
    journal={arXiv preprint arXiv:2411.07228},
    year={2024}
}
```



