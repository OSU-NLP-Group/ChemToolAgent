# ChemToolAgent

This is the official code repo for the paper *ChemToolAgent: The Impact of Tools on Language Agents for Chemistry Problem Solving*. 

🚀 Most tools in ChemToolAgent are updated and released in a standalone toolkit -- [ChemMCP](https://osu-nlp-group.github.io/ChemMCP). It is an easy-to-use and extensible toolkit, compatible with MCP. We will continue to maintain and add more tools there. Join us and let's build more tools and agents together!

**News:**

- [2025.05] We release the ChemToolAgent tools in [ChemMCP](https://osu-nlp-group.github.io/ChemMCP), which is a continuously updated and MCP-compatible toolkit.
- [2025.03] Our [new arXiv version](https://arxiv.org/abs/2411.07228) now includes experimental results on SciBench. Check it out!
- [2025.03] ChemAgent is now **ChemToolAgent**, and the paper title is also updated (previously *Tooling or Not Tooling? The Impact of Tools on Language Agents for Chemistry Problem Solving*).
- [2025.01] Our work is accepted to NAACL 2025 Findings.

## Installation

**Download Checkpoints**

Please download the checkpoints for the property prediction tools from [here](https://zenodo.org/records/15299461). Unzip and put it at `chemagent/tools/property_prediction/checkpoints`.

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
@article{yu2024chemtoolagent,
    title={ChemToolAgent: The Impact of Tools on Language Agents for Chemistry Problem Solving},
    author={Botao Yu and Frazier N. Baker and Ziru Chen and Garrett Herb and Boyu Gou and Daniel Adu-Ampratwum and Xia Ning and Huan Sun},
    journal={arXiv preprint arXiv:2411.07228},
    year={2024}
}
```



