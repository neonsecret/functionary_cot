# Please install transformers & llama-cpp-python to run this script.
# This script is an example of inference using llama-cpp-python + HF tokenizer
from llama_cpp import Llama

from functionary.prompt_template import get_prompt_template_from_tokenizer, get_prompt_template_by_version
from transformers import AutoTokenizer

if __name__ == '__main__':
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g., San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # You can download gguf files from https://huggingface.co/meetkai/functionary-small-v2.5-GGUF
    llm = Llama.from_pretrained(
        repo_id="bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        filename="*Q4_K_M*",
        verbose=False,
        n_ctx=1024
    )
    messages = [{"role": "user", "content": "what's the weather like in Hanoi?"}]

    # Create tokenizer from HF. We should use tokenizer from HF to make sure that tokenizing is correct
    # Because there might be a mismatch between llama-cpp tokenizer and HF tokenizer and the model was trained using HF tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", legacy=True
    )
    # prompt_template will be used for creating the prompt
    # prompt_template = get_prompt_template_from_tokenizer(tokenizer)
    prompt_template = get_prompt_template_by_version("v3-llama3.1-deepseek-r1-think")

    # Before inference, we need to add an empty assistant (message without content or function_call)
    messages.append({"role": "assistant"})

    # Create the prompt to use for inference
    prompt_str = prompt_template.get_prompt_from_messages(messages, tools)
    token_ids = tokenizer.encode(prompt_str)

    gen_tokens = []
    # Get list of stop_tokens
    stop_token_ids = [
        tokenizer.encode(token)[-1]
        for token in prompt_template.get_stop_tokens_for_generation()
    ]

    # We use function generate (instead of __call__) so we can pass in list of token_ids
    for token_id in llm.generate(token_ids):
        if token_id in stop_token_ids:
            break
        gen_tokens.append(token_id)

    llm_output = tokenizer.decode(gen_tokens)
    print(llm_output)

    # parse the message from llm_output
    result = prompt_template.parse_assistant_response(llm_output)
    print(result)
