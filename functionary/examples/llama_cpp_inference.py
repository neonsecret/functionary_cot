# Please install transformers & llama-cpp-python to run this script
# This script is an example of inference using llama-cpp-python + HF tokenizer
from llama_cpp import Llama

from functionary.prompt_template import get_prompt_template_by_version, get_prompt_template_from_tokenizer
from transformers import AutoTokenizer

if __name__ == '__main__':
    tools = [{'type': 'function', 'function': {'name': 'get_stock_price', 'description': 'Get the current stock price',
                                               'parameters': {'type': 'object', 'properties': {
                                                   'symbol': {'type': 'string', 'description': 'The stock symbol'}},
                                                              'required': ['symbol']}}}]

    # tools = [
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "get_current_weather",
    #             "description": "Get the current weather",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "location": {
    #                         "type": "string",
    #                         "description": "The city and state, e.g., San Francisco, CA",
    #                     }
    #                 },
    #                 "required": ["location"],
    #             },
    #         },
    #     },
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "get_time",
    #             "description": "Get the current Time",
    #             "properties": {},
    #             "parameters": {
    #                 "required": [],
    #             },
    #         },
    #     }
    # ]

    # You can download gguf files from https://huggingface.co/meetkai/functionary-small-v2.5-GGUF
    PATH_TO_GGUF_FILE = "out_gguf/out_my_t_5e-Q4_K_M.gguf"
    llm = Llama(model_path=PATH_TO_GGUF_FILE, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    messages = [{"role": "user", "content": "Can you tell me the current stock price of Apple? symbol is AAPL"}]

    # Create tokenizer from HF. We should use tokenizer from HF to make sure that tokenizing is correct
    # Because there might be a mismatch between llama-cpp tokenizer and HF tokenizer and the model was trained using HF tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", legacy=True
    )
    # prompt_template will be used for creating the prompt
    prompt_template = get_prompt_template_by_version("v3-llama3.1-deepseek-r1-think")

    # Before inference, we need to add an empty assistant (message without content or function_call)
    messages.append({"role": "assistant"})

    # Create the prompt to use for inference
    prompt_str = prompt_template.get_prompt_from_messages(messages, tools)
    print("\n prompt str: ", prompt_str)
    token_ids = tokenizer.encode(prompt_str)

    gen_tokens = []
    # Get list of stop_tokens
    stop_token_ids = [
        tokenizer.encode(token)[-1]
        for token in prompt_template.get_stop_tokens_for_generation()
    ]

    # We use function generate (instead of __call__) so we can pass in list of token_ids
    for token_id in llm.generate(token_ids, temp=0.2):
        if token_id in stop_token_ids:
            break
        gen_tokens.append(token_id)

    llm_output = tokenizer.decode(gen_tokens)
    print("\n", llm_output, "\n")

    # parse the message from llm_output
    result = prompt_template.parse_assistant_response(llm_output)
    print("\n Result: ", result, "\n")
