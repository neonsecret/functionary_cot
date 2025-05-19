import datetime
import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.base_template import PromptTemplate
from functionary.prompt_template.llama31_prompt_template import Llama31Template, parse_function_call_from_text

THINK_BLOCK_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


class Llama31DeepseekR1ThinkTemplate(PromptTemplate):
    version = "v3-llama3.1-deepseek-r1-think"

    function_separator = Llama31Template.function_separator
    start_header = Llama31Template.start_header
    end_header = Llama31Template.end_header
    eos_token = "<｜end▁of▁sentence｜>"
    eof_message = Llama31Template.eof_message
    fn_param_sep_token = Llama31Template.fn_param_sep_token

    def get_additional_tokens(self) -> List[str]:

        return []

    def get_assistant_prefixes(self) -> List[str]:

        return ["<｜Assistant｜>"]

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids"""

        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.eos_token]

    def parse_assistant_response(
            self, llm_output: str, tool_choice: Any = None
    ) -> Dict:
        """
        Parses the LLM output string into a dictionary.
        Handles the <think> block and Llama 3.1 style function calls.
        """
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        original_llm_output = llm_output
        think_content = None
        think_match = THINK_BLOCK_PATTERN.match(llm_output)
        if think_match:
            think_content = think_match.group(0).strip()

            llm_output = llm_output[think_match.end():]

        else:
            pass

        llm_output = (
                self.get_generation_prefix_for_tool_choice(tool_choice) + llm_output
        )

        tool_calls = []
        text_response = ""

        func_prefix = "<function="
        end_func = "</function>"
        python_tag = "<|python_tag|>"

        current_pos = 0
        while current_pos < len(llm_output):
            remaining_output = llm_output[current_pos:]

            if remaining_output.startswith(python_tag):
                if current_pos > 0 and text_response == "":
                    text_response = llm_output[:current_pos].strip()

                code = remaining_output[len(python_tag):]
                function_call = {
                    "name": "python",
                    "arguments": code,
                }
                tool_calls.append(
                    {
                        "type": "function",
                        "id": prompt_utils.get_random_tool_call_id(),
                        "function": function_call,
                    }
                )
                current_pos = len(llm_output)


            elif remaining_output.startswith(func_prefix):

                if current_pos > 0 and text_response == "":
                    text_response = llm_output[:current_pos].strip()

                end_index = remaining_output.find(end_func)
                if end_index != -1:

                    full_call_block = remaining_output[:end_index + len(end_func)]

                    call_content = full_call_block[len(func_prefix):end_index]

                    function_call = parse_function_call_from_text(call_content)
                    if function_call:
                        tool_calls.append(
                            {
                                "type": "function",
                                "id": prompt_utils.get_random_tool_call_id(),
                                "function": function_call,
                            }
                        )
                    else:
                        print(f"WARN: Could not parse function call content: {call_content}")
                        text_response += full_call_block

                    current_pos += len(full_call_block)
                else:

                    print(f"WARN: Found '{func_prefix}' but no closing '{end_func}'. Treating rest as text.")
                    text_response += remaining_output
                    current_pos = len(llm_output)
            else:

                current_pos += 1

        if not tool_calls and current_pos == len(llm_output) and text_response == "":
            text_response = llm_output.strip()

        if not text_response or text_response.strip() == "":
            text_response = None
        if not tool_calls:
            tool_calls = None

        if tool_calls and text_response is not None and text_response.strip() == "":
            text_response = None

        if text_response is not None and tool_calls is not None:
            print(
                f"WARN: Parsed both text content and tool calls. Text: '{text_response[:50]}...', Tool Calls: {len(tool_calls)}")

        final_content = None
        if think_content:
            final_content = think_content
            if text_response and text_response.strip():
                final_content += "\n" + text_response.strip()

        elif text_response and text_response.strip():
            final_content = text_response.strip()
        return {"role": "assistant", "content": final_content, "tool_calls": tool_calls}

    def initialize_fsm_gen_state(
            self,
            tool_choice: Union[str, Tool],
            curr_text: str,
            curr_tokens: Optional[List[int]],
            add_code_interpreter: Optional[bool],
    ) -> Dict:
        """Initializes the FSM state, adding tracking for the think block."""

        base_gen_state = super(Llama31Template, self).initialize_fsm_gen_state(
            tool_choice=tool_choice,
            curr_text=curr_text,
            curr_tokens=curr_tokens,
            add_code_interpreter=add_code_interpreter,
        )

        base_gen_state["is_in_think_block"] = True
        base_gen_state["think_buffer"] = ""
        base_gen_state["processed_after_think"] = False

        if base_gen_state["is_in_think_block"]:
            pass

        print(
            f"DEBUG: Initial FSM state: stage={base_gen_state['stage']}, is_in_think_block={base_gen_state['is_in_think_block']}")
        return base_gen_state

    def stream_delta_text(
            self,
            gen_state: Dict,
            delta_text: str,
            finish_reason: Optional[str],
            tools_or_functions: List[Dict],
            tool_choice: Any,
    ) -> Tuple[Dict, Optional[Union[Dict, List[Dict]]]]:
        """
        Generates delta messages for streaming, skipping output during the think block.
        """
        if finish_reason is not None:

            final_finish_reason = finish_reason
            if gen_state["stage"] in ["parameter", "code-interpreter"] and not gen_state["is_in_think_block"]:
                final_finish_reason = "tool_calls"
            elif gen_state["is_in_think_block"]:
                print("WARN: Finish reason received while FSM state indicates still in think block.")
                if gen_state["processed_after_think"] and gen_state["stage"] in ["parameter", "code-interpreter"]:
                    final_finish_reason = "tool_calls"
                else:
                    final_finish_reason = "stop"

            return gen_state, prompt_utils.get_text_delta_response(
                None, False, final_finish_reason
            )

        if gen_state["is_in_think_block"]:
            print(f"DEBUG: In think block, skipping delta generation for: '{delta_text}'")
            gen_state = self.update_fsm_gen_state(gen_state, delta_text, None, [], None)
            if not gen_state["is_in_think_block"] and gen_state["processed_after_think"]:
                return gen_state, []
            else:
                return gen_state, []

        options = self.get_options_from_gen_state(
            gen_state=gen_state, tools_or_functions=tools_or_functions
        )

        responses = []
        if gen_state["stage"] == "text-gen":

            if gen_state["first_chunk"]:
                responses.append(
                    prompt_utils.get_text_delta_response("", True, finish_reason)
                )
                gen_state["first_chunk"] = False
            responses.append(
                prompt_utils.get_text_delta_response(
                    delta_text, False, finish_reason
                )
            )

        elif gen_state["stage"] == "parameter":

            if gen_state["first_function_chunk"]:
                responses.append(
                    prompt_utils.get_function_delta_response(
                        gen_state, "", True, gen_state["first_chunk"], finish_reason
                    )
                )
                gen_state["first_chunk"] = False
                gen_state["first_function_chunk"] = False

                if gen_state["curr_text"] != "":
                    responses.append(
                        prompt_utils.get_function_delta_response(
                            gen_state, gen_state["curr_text"], False, False, finish_reason
                        )
                    )
            if "</" not in delta_text and "</" not in gen_state["curr_text"]:
                responses.append(
                    prompt_utils.get_function_delta_response(
                        gen_state, delta_text, False, False, finish_reason
                    )
                )
            elif "</" in delta_text:
                suffix_match = re.search(r"</[^>]*$", delta_text)
                prefix = delta_text[:suffix_match.start()] if suffix_match else delta_text
                if prefix:
                    responses.append(
                        prompt_utils.get_function_delta_response(
                            gen_state, prefix, False, False, finish_reason
                        )
                    )

        elif gen_state["stage"] == "code-interpreter":
            if gen_state["first_function_chunk"]:
                responses.append(
                    prompt_utils.get_function_delta_response(
                        gen_state, "", True, gen_state["first_chunk"], finish_reason
                    )
                )
                gen_state["first_chunk"] = False
                gen_state["first_function_chunk"] = False
            responses.append(
                prompt_utils.get_function_delta_response(
                    gen_state, delta_text, False, False, finish_reason
                )
            )
        elif gen_state["stage"] == "function":
            pass
        elif gen_state["stage"] == "pre-function":
            pass

        gen_state = self.update_fsm_gen_state(
            gen_state=gen_state,
            new_token=delta_text,
            new_token_id=None,
            options=options,
            tokenizer=None,
        )

        return gen_state, responses

    def update_fsm_gen_state(
            self,
            gen_state: Dict,
            new_token: Optional[str],
            new_token_id: Optional[int],
            options: Optional[List],
            tokenizer: Any,
    ) -> Dict:
        """Updates the FSM state, handling the transition out of the think block."""

        current_full_text = gen_state["curr_text"] + (new_token if new_token else "")
        gen_state["curr_text"] = current_full_text

        if gen_state["is_in_think_block"]:
            gen_state["think_buffer"] += (new_token if new_token else "")

            if "</think>" in gen_state["think_buffer"]:
                print(f"DEBUG: Detected '</think>' in buffer: '{gen_state['think_buffer'][-20:]}'")
                gen_state["is_in_think_block"] = False

                think_end_match = re.search(r"</think.*?>", gen_state["think_buffer"], re.IGNORECASE)
                if think_end_match:
                    post_think_text = gen_state["think_buffer"][think_end_match.end():]
                    gen_state["curr_text"] = post_think_text.lstrip()
                    print(f"DEBUG: Exited think block. Reset curr_text to: '{gen_state['curr_text'][:100]}...'")

                    gen_state["think_buffer"] = ""
                    gen_state["processed_after_think"] = True

                    gen_state["first_chunk"] = True
                    gen_state["first_function_chunk"] = True

                    if gen_state["curr_text"].startswith("<|python_tag|>"):
                        gen_state = self._update_gen_state_for_fn_call(gen_state, "python")
                        gen_state["stage"] = "code-interpreter"
                        gen_state = self._reset_fsm_curr_text_and_tokens(
                            gen_state)
                    elif gen_state["curr_text"].startswith("<function="):
                        gen_state["stage"] = "function"

                        match = re.match(r"<function=([^>]+)>", gen_state["curr_text"])
                        if match:
                            func_name = match.group(1)
                            gen_state = self._update_gen_state_for_fn_call(gen_state, func_name)
                            gen_state["stage"] = "parameter"

                            gen_state["curr_text"] = gen_state["curr_text"][match.end():]
                    elif gen_state["curr_text"].startswith("<"):

                        gen_state["stage"] = "pre-function"
                    elif len(gen_state["curr_text"]) > 0:

                        gen_state["stage"] = "text-gen"
                    else:

                        gen_state["stage"] = "pre-function"

                    print(f"DEBUG: Post-think state transition complete. New stage: {gen_state['stage']}")

                else:

                    print("WARN: Detected '</think>' but regex failed to find end tag.")

            return gen_state

        stage = gen_state["stage"]
        curr_text = gen_state["curr_text"]

        if stage == "pre-function":

            if curr_text.endswith("<|python_tag|>"):
                gen_state = self._update_gen_state_for_fn_call(gen_state, "python")
                gen_state["stage"] = "code-interpreter"
                gen_state = self._reset_fsm_curr_text_and_tokens(gen_state)
            elif curr_text.endswith("<function="):
                gen_state["stage"] = "function"
            elif not curr_text.startswith("<"):
                gen_state["stage"] = "text-gen"

        elif stage == "text-gen":

            if "<|python_tag|>" in curr_text:

                gen_state = self._update_gen_state_for_fn_call(gen_state, "python")
                gen_state["stage"] = "code-interpreter"
                gen_state = self._reset_fsm_curr_text_and_tokens(gen_state)
            elif "<function=" in curr_text:

                gen_state["stage"] = "function"

                func_start_index = curr_text.find("<function=")
                gen_state["curr_text"] = curr_text[func_start_index:]

        elif stage == "function":

            pattern = r"<function=([^>]+)>"
            match = re.search(pattern, curr_text)
            if match:
                func_name = match.group(1)
                gen_state = self._update_gen_state_for_fn_call(gen_state, func_name)
                gen_state["stage"] = "parameter"

                gen_state["curr_text"] = curr_text[match.end():]
                gen_state["first_function_chunk"] = True

        elif stage == "parameter":

            if "</function>" in curr_text:
                gen_state["stage"] = "pre-function"

                end_tag_index = curr_text.rfind("</function>")
                gen_state["curr_text"] = curr_text[end_tag_index + len("</function>"):]

                gen_state["first_chunk"] = True
                gen_state["first_function_chunk"] = True

        return gen_state

    def _update_gen_state_for_fn_call(self, gen_state: Dict, func_name: str) -> Dict:
        """Updates gen_state when a function call is identified."""
        gen_state["func_name"] = func_name
        gen_state["func_index"] += 1
        gen_state["call_id"] = f"{prompt_utils.TOOL_CALL_ID_PREFIX}-{gen_state['func_index']}"

        gen_state["first_function_chunk"] = True
        return gen_state

    def _reset_fsm_curr_text_and_tokens(self, gen_state: Dict) -> Dict:
        """Resets current text and tokens in FSM state."""
        gen_state["curr_text"] = ""
        gen_state["curr_tokens"] = []
        return gen_state

    def get_options_from_gen_state(self, gen_state: Dict, tools_or_functions: List):

        return []

    def get_chat_template_jinja(self):
        return super().get_chat_template_jinja()

    def get_force_function_call_prefix(self, function_name: str):

        return f"<function={function_name}>"

    def get_force_text_generation_prefix(self):

        return f""

    def get_tool_choice_required_prefix(self):

        return "<function="

    def get_generation_prefix_for_tool_choice(self, tool_choice: Any) -> str:
        """Gets the prefix based on tool_choice constraint for generation"""

        prefix = ""
        if tool_choice == "none":
            prefix = self.get_force_text_generation_prefix()
        elif tool_choice == "required" or tool_choice == "auto":

            if tool_choice == "required":
                prefix = self.get_tool_choice_required_prefix()
        elif isinstance(tool_choice, (Tool, Function)) or (isinstance(tool_choice, dict) and "function" in tool_choice):

            func_name = tool_choice.function.name if hasattr(tool_choice, 'function') else tool_choice.get('function',
                                                                                                           {}).get(
                'name')
            if func_name:
                prefix = self.get_force_function_call_prefix(func_name)
        return prefix
