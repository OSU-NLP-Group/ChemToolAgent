import os

from chemagent.tools import *


ALL_TOOL_NAMES = {
    'PubchemSearchQA',
    'IUPAC2SMILES',
    'SMILES2IUPAC',
    'Name2SMILES',
    'SMILES2SELFIES',
    'SELFIES2SMILES',
    'SMILES2Formula',
    'PatentCheck',
    'CanonicalizeSMILES',
    'CompareSMILES',
    'CountMolAtoms',
    'MolSimilarity',
    'SMILES2Weight',
    'FunctionalGroups',
    'GetMoleculePrice',
    'WikipediaSearch',
    'PythonREPL',
    'MoleculeCaptioner',
    'MoleculeGenerator',
    'ForwardSynthesis',
    'Retrosynthesis',
    'WebSearch',
    'AiExpert',
    'SolubilityPredictor',
    'LogDPredictor',
    'BBBPPredictor',
    'ToxicityPredictor',
    'HIVInhibitorPredictor',
    'SideEffectPredictor'
}


def make_tools(llm, api_keys: dict = {}, init=True, include_tools=None, exclude_tools=None):
    tavily_api_key = api_keys.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
    rxn4chem_api_key = api_keys.get("RXN4CHEM_API_KEY") or os.getenv("RXN4CHEM_API_KEY")
    openai_api_key = api_keys.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    anthropic_api_key = api_keys.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    chemspace_api_key = api_keys.get("CHEMSPACE_API_KEY") or os.getenv(
        "CHEMSPACE_API_KEY"
    )

    all_tools = []

    all_tools += [
        PubchemSearchQA(api_keys={'OPENAI_API_KEY': openai_api_key, 'ANTHROPIC_API_KEY': anthropic_api_key}, init=init),
        IUPAC2SMILES(chemspace_api_key, init=init),
        SMILES2IUPAC(init=init),
        Name2SMILES(init=init),
        SMILES2SELFIES(init=init),
        SELFIES2SMILES(init=init),
        SMILES2Formula(init=init),
        PatentCheck(init=init),
        CanonicalizeSMILES(init=init),
        CompareSMILES(init=init),
        CountMolAtoms(init=init),
        MolSimilarity(init=init),
        SMILES2Weight(init=init),
        FuncGroups(init=init),
        GetMoleculePrice(chemspace_api_key, init=init),
        Wikipedia(init=init),
        PropertyPredictorESOL(init=init),
        PropertyPredictorLIPO(init=init),
        PropertyPredictorBBBP(init=init),
        PropertyPredictorClinTox(init=init),
        PropertyPredictorHIV(init=init),
        PropertyPredictorSIDER(init=init),
        PythonShell(init=init),
        MoleculeCaptioner(init=init),
        MoleculeGenerator(init=init),
    ]
    if rxn4chem_api_key:
        all_tools += [
            ForwardSynthesis(rxn4chem_api_key, init=init),
            Retrosynthesis(rxn4chem_api_key, init=init),
        ]
    if tavily_api_key:
        all_tools += [WebSearch(tavily_api_key, init=init)]
    all_tools += [
        AiExpert(api_keys={'OPENAI_API_KEY': openai_api_key, 'ANTHROPIC_API_KEY': anthropic_api_key}, model=llm, init=init)
    ]

    final_tools = []
    if include_tools is not None:
        include_tools = set(include_tools)
        assert exclude_tools is None

        for tool in all_tools:
            if tool.name in include_tools:
                final_tools.append(tool)
    elif exclude_tools is not None:
        exclude_tools = set(exclude_tools)
        assert include_tools is None

        for tool in all_tools:
            if tool.name not in exclude_tools:
                final_tools.append(tool)
    else:
        final_tools = all_tools

    return final_tools


def verify_tools(tools):
    # Referring to ALL_TOOL_NAMES, check if all tools are included in the list of tools
    # The tool name can be obtained with tool.name
    # Return three lists: missing_tools, extra_tools, and tools that are included more than once
    duplicate_tools = set()
    tool_names = set()
    for tool in tools:
        name = tool.name
        if name in tool_names:
            duplicate_tools.add(name)
        else:
            tool_names.add(name)
    missing_tools = ALL_TOOL_NAMES - tool_names
    extra_tools = tool_names - ALL_TOOL_NAMES
    return missing_tools, extra_tools, duplicate_tools


def make_code_tools(llm, api_keys: dict = {}, init=True):
    tavily_api_key = api_keys.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
    rxn4chem_api_key = api_keys.get("RXN4CHEM_API_KEY") or os.getenv("RXN4CHEM_API_KEY")
    openai_api_key = api_keys.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    anthropic_api_key = api_keys.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    chemspace_api_key = api_keys.get("CHEMSPACE_API_KEY") or os.getenv(
        "CHEMSPACE_API_KEY"
    )

    all_tools = []

    all_tools += [
        PubchemSearchQA(api_keys={'OPENAI_API_KEY': openai_api_key, 'ANTHROPIC_API_KEY': anthropic_api_key}, init=init, interface='code'),
        IUPAC2SMILES(chemspace_api_key, init=init, interface='code'),
        SMILES2IUPAC(init=init, interface='code'),
        SMILES2SELFIES(init=init, interface='code'),
        SELFIES2SMILES(init=init, interface='code'),
        SMILES2Formula(init=init, interface='code'),
        PatentCheck(init=init, interface='code'),
        CanonicalizeSMILES(init=init, interface='code'),
        CompareSMILES(init=init, interface='code'),
        CountMolAtoms(init=init, interface='code'),
        MolSimilarity(init=init, interface='code'),
        SMILES2Weight(init=init, interface='code'),
        FuncGroups(init=init, interface='code'),
        GetMoleculePrice(chemspace_api_key, init=init, interface='code'),
        Wikipedia(init=init, interface='code'),
        PythonShell(init=init, interface='code'),
        MoleculeCaptioner(init=init, interface='code'),
        MoleculeGenerator(init=init, interface='code'),
    ]
    if rxn4chem_api_key:
        all_tools += [
            ForwardSynthesis(rxn4chem_api_key, init=init, interface='code'),
            Retrosynthesis(rxn4chem_api_key, init=init, interface='code'),
            # RXNRetrosynthesis(rxn4chem_api_key, openai_api_key, init=init),
        ]
    if tavily_api_key:
        all_tools += [WebSearch(tavily_api_key, init=init, interface='code')]
    if openai_api_key:
        all_tools += [
            AiExpert(api_keys={'OPENAI_API_KEY': openai_api_key, 'ANTHROPIC_API_KEY': anthropic_api_key}, model=llm, init=init, interface='code')
        ]

    return all_tools


def generate_code_tools_description(tools):
    descriptions = {}
    for tool in tools:
        func_name = tool.__class__.func_name
        func_doc = tool.__class__.func_doc
        func_definition = '%s(%s) -> %s' % (
            func_name,
            ', '.join([item for item in func_doc[:-1]]),
            func_doc[-1],
        )
        func_description = tool.__class__.func_description
        item = (func_definition, func_description)
        descriptions[func_name] = item
    return descriptions
