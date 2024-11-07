import openai
from .openai_llm import GptRequester, NewGptRequester
from .anthropic_llm import ClaudeRequester


def make_llm(model, api_keys, **kwargs):
    if model.startswith("gpt") or model.startswith('o1'):
        api_key = api_keys['OPENAI_API_KEY']
        if openai.__version__.startswith('0.'):
            llm = GptRequester(api_code=api_key, model_name=model, **kwargs)
        else:
            llm = NewGptRequester(api_code=api_key, model_name=model, **kwargs)
    elif model.startswith("claude"):
        api_key = api_keys['ANTHROPIC_API_KEY']
        llm = ClaudeRequester(api_code=api_key, model_name=model, **kwargs)
    else:
        raise NotImplementedError(f"Support for model {model} not implemented.")
    return llm
