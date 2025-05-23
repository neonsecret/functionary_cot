"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Conversion between OpenAI APIs and native SRT APIs"""

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple, Union, Any

import sglang as sgl
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from sglang.lang.choices import greedy_token_selection
from sglang.lang.interpreter import ProgramState
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from transformers import AutoTokenizer

from functionary.inference_stream import generate_openai_format_from_stream_async
from functionary.inference_utils import (
    analyze_tools_and_tool_choice,
    check_all_errors,
    convert_tool_calls_to_function_call,
    create_error_response,
)
from functionary.openai_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    Function,
    StreamChoice,
    Tool,
    UsageInfo,
)
from functionary.prompt_template import (
    PromptTemplate,
    get_prompt_template_from_tokenizer,
)
from functionary.prompt_template.prompt_utils import prepare_messages_for_inference

# Choices sampling method for sgl.select
CHOICES_SAMPLING_METHOD = greedy_token_selection
# Variable name for sgl frontend runtime generation
CONTENT_VAR = "content"


@dataclass
class ChatCompletionParams:
    """Parameters and context used across various chat completion functions"""

    adapted_request: GenerateReqInput
    raw_request: Request
    request: ChatCompletionRequest
    tokenizer: AutoTokenizer
    tokenizer_manager: Optional[TokenizerManager]
    srt_backend: Any
    prompt_template: PromptTemplate
    tools_or_functions: List[Dict]
    tool_func_choice: Optional[Union[str, Tool, Function]]
    frontend_state: Optional[ProgramState]
    grammar_sampling: bool


def v1_chat_generate_request(
    request: ChatCompletionRequest,
    tokenizer: AutoTokenizer,
    tools_or_functions: List[Dict],
    tool_func_choice: Optional[Union[str, Tool, Function]],
    return_text: bool = False,
) -> Tuple[GenerateReqInput, ChatCompletionRequest]:
    """
    Generate an adapted request that SGLang uses.

    This function prepares the input for SGLang inference by processing the chat completion request,
    applying the appropriate tokenization, and setting up the sampling parameters.

    Args:
        request (ChatCompletionRequest): The original chat completion request.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the text input, if any.
        tools_or_functions (List[Dict]): List of available tools or functions.
        tool_func_choice (Optional[Union[str, Tool, Function]]): The chosen tool or function, if any.
        return_text (bool, optional): Whether to return the input as text instead of token IDs. Defaults to False.

    Returns:
        Tuple[GenerateReqInput, ChatCompletionRequest]: A tuple containing:
            - The adapted request (GenerateReqInput) to be used by SGLang.
            - The original request (ChatCompletionRequest), NOT modified.

    Note:
        This function handles the conversion of the chat messages into a format suitable for SGLang,
        applies the chat template, sets up stopping criteria, and configures sampling parameters.
    """
    # Apply chat template and its stop strings.
    input_ids = prepare_messages_for_inference(
        tokenizer=tokenizer,
        messages=request.messages,
        tools_or_functions=tools_or_functions,
        tool_choice=tool_func_choice,
        device="cpu",
        return_text=return_text,
    )
    if not return_text:
        input_ids = input_ids.tolist()[0]

    stop = (
        request.stop
        + get_prompt_template_from_tokenizer(
            tokenizer=tokenizer
        ).get_stop_tokens_for_generation()
    )
    sampling_params = {
        "temperature": request.temperature,
        "max_new_tokens": request.max_tokens,
        "min_new_tokens": request.min_tokens,
        "stop": stop,
        "stop_token_ids": request.stop_token_ids,
        "top_p": request.top_p,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "repetition_penalty": request.repetition_penalty,
        "regex": request.regex,
        "n": request.n,
        "skip_special_tokens": False,
    }

    if isinstance(input_ids, str):
        prompt_kwargs = {"text": input_ids}
    else:
        prompt_kwargs = {"input_ids": input_ids}

    adapted_request = GenerateReqInput(
        **prompt_kwargs,
        image_data=None,
        sampling_params=sampling_params,
        return_logprob=request.logprobs,
        top_logprobs_num=request.top_logprobs,
        stream=request.stream,
        return_text_in_logprobs=True,
    )

    return adapted_request, request


async def wrap_sgl_generator(params: ChatCompletionParams):
    """
    This asynchronous generator function yields generated text chunks along
    with their finish reasons.

    Args:
        params (ChatCompletionParams): A dataclass containing all necessary
            parameters for the chat completion, including the request details,
            tokenizer, backend, and other configuration options.

    Yields:
        Tuple[str, Optional[str]]: A tuple containing:
            - str: The generated text chunk.
            - Optional[str]: The finish reason, if any (e.g., "stop", "length", etc.).
    """
    # Iterates over the text generated by the tokenizer manager
    stream_buffer = ""
    async for content in params.tokenizer_manager.generate_request(
        params.adapted_request, params.raw_request
    ):
        text = content["text"]
        delta = text[len(stream_buffer) :]
        stream_buffer = stream_buffer + delta
        finish_reason = content["meta_info"]["finish_reason"]

        # If finish_reason is not None and delta_text is not empty,
        # the delta_text is the eos_token and just remove it
        if finish_reason is not None:
            finish_reason = finish_reason["type"]
            if len(delta) > 0:
                delta = ""
        yield delta, finish_reason


async def completion_stream_generator(params: ChatCompletionParams):
    """
    This asynchronous generator function produces a stream of ChatCompletionChunk
    objects. It handles both grammar-sampling and regular generations,
    depending on the parameters provided.

    Args:
        params (ChatCompletionParams): A dataclass containing all necessary
            parameters for the chat completion, including the request details,
            tokenizer, backend, and other configuration options.

    Yields:
        str: JSON-formatted strings representing chunks of the chat completion
             response, including delta updates and finish reasons.

    Notes:
        - The function adapts its behavior based on whether grammar sampling
          is enabled or not.
        - It handles the conversion of tool calls to function calls when
          appropriate.
        - The stream is terminated with a "[DONE]" message.
    """
    # Initialize the text generator
    generator = wrap_sgl_generator(params)

    tool_call_count = 0
    # Generate the text in openai format
    async for response in generate_openai_format_from_stream_async(
        generator,
        params.prompt_template,
        params.tool_func_choice,
        params.tools_or_functions,
    ):
        # Convert tool_calls to function_call if request.functions is provided
        response = convert_tool_calls_to_function_call(
            functions=params.request.functions, chat_message=response
        )
        if response["delta"]["function_call"]:
            tool_name = response["delta"]["function_call"]["name"]
            tool_args = response["delta"]["function_call"]["arguments"]
            if tool_name and len(tool_name) > 0 and tool_args == "":
                tool_call_count += 1

            # Return finish_reason after the first tool_call is streamed if functions is provided
            if params.request.functions and tool_call_count == 2:
                response["delta"] = {}
                response["finish_reason"] = "function_call"

        chunk = StreamChoice(**response)
        result = ChatCompletionChunk(
            id=params.adapted_request.rid, choices=[chunk], model=params.request.model
        )
        chunk_dic = result.model_dump()
        chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
        yield f"data: {chunk_data}\n\n"
        # Break from for loop after the first tool_call is streamed if functions is provided
        if params.request.functions and tool_call_count == 2:
            break
    yield "data: [DONE]\n\n"


async def v1_chat_generate_completion(
    params: ChatCompletionParams,
) -> Tuple[Union[StreamingResponse, str, List[str]], Optional[JSONResponse]]:
    """
    Generate a text completion.

    This function handles both streaming and non-streaming responses for chat completions.
    It supports both regular and grammar-sampling generations.

    Args:
        params (ChatCompletionParams): A dataclass containing all necessary parameters and context
                                       for generating the text.

    Returns:
        Tuple[Union[StreamingResponse, str], Optional[JSONResponse]]:
            - If streaming is requested, returns a StreamingResponse object.
            - If non-streaming, returns the generated text as a string.
            - The second element is an optional JSONResponse for error cases.

    Note:
        - For grammar-sampling, it uses the SGLang Frontend Runtime.
        - For regular generation, it uses the tokenizer manager to generate the response.
        - Streaming responses are handled by the completion_stream_generator function.
    """
    # If streaming, return the StreamingResponse else return the text
    if params.adapted_request.stream:
        return (
            StreamingResponse(
                completion_stream_generator(params),
                media_type="text/event-stream",
                background=params.tokenizer_manager.create_abort_task(
                    params.adapted_request
                ),
            ),
            None,
        )
    else:
        try:
            ret = await params.tokenizer_manager.generate_request(
                params.adapted_request, params.raw_request
            ).__anext__()
        except ValueError as e:
            return None, create_error_response(
                status_code=HTTPStatus.BAD_REQUEST, message=str(e), param=None
            )
        if (
            type(ret) == list
        ):  # if n > 1 (multiple samples), we return a list of strings
            return [item["text"] for item in ret], None
        else:
            return ret["text"], None


def v1_chat_generate_response(
    output_text: Union[str, List[str]], params: ChatCompletionParams
) -> ChatCompletionResponse:
    """
    Generate a ChatCompletionResponse from the output text and parameters.

    This function processes the output text, parses it according to the prompt template,
    and constructs a ChatCompletionResponse object.

    Args:
        output_text (str): The raw output text from SGLang inference.
        params (ChatCompletionParams): Parameters and context for the chat completion.

    Returns:
        ChatCompletionResponse: An OpenAI-compatible response containing the assistant's message,
        usage information, and other metadata.
    """
    output_texts = output_text if type(output_text) == list else [output_text]
    choices = []
    prompt_tokens, completion_tokens = 0, 0
    # Parse the output text using the specific prompt template
    for text in output_texts:
        chat_mess = params.prompt_template.parse_assistant_response(
            llm_output=text, tool_choice=params.tool_func_choice
        )
        # Convert tool_calls to function_call if request.functions is provided
        chat_mess = convert_tool_calls_to_function_call(
            functions=params.request.functions, chat_message=chat_mess
        )

        # Postprocess finish reason
        finish_reason = "stop"
        if params.tool_func_choice is None or params.tool_func_choice in [
            "auto",
            "required",
        ]:
            if "function_call" in chat_mess and chat_mess["function_call"]:
                finish_reason = "function_call"
            if "tool_calls" in chat_mess and chat_mess["tool_calls"]:
                finish_reason = "tool_calls"

        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(**chat_mess),
                finish_reason=finish_reason,
            )
        )

        prompt_tokens += (
            len(params.adapted_request.input_ids)
            if params.adapted_request.input_ids
            else len(params.tokenizer.encode(params.adapted_request.text))
        )
        completion_tokens += (
            len(params.tokenizer.encode(text, add_special_tokens=False)) + 1
        )  # +1 for the eos token

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        model=params.request.model,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return response


async def v1_chat_completions(
    tokenizer_manager: Optional[TokenizerManager],
    srt_backend: Any,
    raw_request: Request,
    served_model: List[str],
):
    """
    Handle chat completions for v1 of the API.

    This function processes the incoming request, prepares the necessary parameters,
    generates the chat completion, and returns the response. It supports both
    streaming and non-streaming responses.

    Args:
        tokenizer_manager (Optional[TokenizerManager]): Manager for tokenization tasks.
            None if grammar sampling is enabled.
        srt_backend (Optional[Runtime]): The SRT backend for processing.
            None if grammar sampling is disabled.
        raw_request (Request): The raw incoming request object.

    Returns:
        Union[ChatCompletionResponse, StreamingResponse, JSONResponse]:
            - ChatCompletionResponse for non-streaming successful responses.
            - StreamingResponse for streaming responses.
            - JSONResponse for error responses.

    Raises:
        No explicit raises, but may return error responses for various failure scenarios.
    """
    request_json = await raw_request.json()
    request = ChatCompletionRequest(**request_json)
    tokenizer = (
        tokenizer_manager.tokenizer
        if tokenizer_manager
        else srt_backend.get_tokenizer()
    )
    prompt_template = get_prompt_template_from_tokenizer(tokenizer=tokenizer)
    tools_or_functions, tool_func_choice = analyze_tools_and_tool_choice(request)

    # Check for errors
    error_check_ret = await check_all_errors(request, served_model)
    if error_check_ret is not None:
        return error_check_ret

    # Generate the adapted request
    adapted_request, request = v1_chat_generate_request(
        request, tokenizer, tools_or_functions, tool_func_choice, return_text=False
    )

    # Prepare the parameters for generate_completion and generate_response functions
    params = ChatCompletionParams(
        adapted_request=adapted_request,
        raw_request=raw_request,
        request=request,
        tokenizer=tokenizer,
        tokenizer_manager=tokenizer_manager,
        srt_backend=srt_backend,
        prompt_template=prompt_template,
        tools_or_functions=tools_or_functions,
        tool_func_choice=tool_func_choice,
        frontend_state=None,  # None first. Set later if needed
        grammar_sampling=True if srt_backend else False,
    )

    # Generate the text completion
    output, error = await v1_chat_generate_completion(params)
    if error:
        return error

    # If streaming, return the output(StreamingResponse) directly
    if adapted_request.stream:
        return output

    # Generate the API response
    response = v1_chat_generate_response(output_text=output, params=params)

    return response
