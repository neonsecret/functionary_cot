# llama31_deepseekr1_think_prompt_template.py

import datetime
import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
# Import the base class and the Llama31Template to potentially reuse methods
from functionary.prompt_template.base_template import PromptTemplate
from functionary.prompt_template.llama31_prompt_template import Llama31Template, parse_function_call_from_text

# Define the think block pattern (allowing for leading/trailing whitespace)
# re.DOTALL allows '.' to match newlines within the think block
THINK_BLOCK_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


class Llama31DeepseekR1ThinkTemplate(PromptTemplate):
    # Change version to match your Jinja template identifier
    version = "v3-llama3.1-deepseek-r1-think"

    function_separator = Llama31Template.function_separator
    start_header = Llama31Template.start_header
    end_header = Llama31Template.end_header
    eos_token = "<｜end▁of▁sentence｜>"
    eof_message = Llama31Template.eof_message
    fn_param_sep_token = Llama31Template.fn_param_sep_token

    def get_additional_tokens(self) -> List[str]:
        # No new structural tokens added beyond Llama 3.1 base
        return []

    def get_assistant_prefixes(self) -> List[str]:
        # Use the DeepSeek R1 Assistant tag
        return ["<｜Assistant｜>"]

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids"""
        # Reuse Llama 3.1's utility function
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
        # First remove stop tokens if they exist
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        # --- MODIFICATION START: Handle <think> block ---
        original_llm_output = llm_output  # Keep for potential debugging
        think_content = None
        think_match = THINK_BLOCK_PATTERN.match(llm_output)
        if think_match:
            think_content = think_match.group(0).strip()
            # Strip the think block from the beginning
            llm_output = llm_output[think_match.end():]
            # print(f"DEBUG: Stripped think block. Remaining output: '{llm_output[:100]}...'")  # Debug print
        else:
            pass
            # print(f"DEBUG: No think block found at start of: '{llm_output[:100]}...'")  # Debug print
        # --- MODIFICATION END ---

        # Add forced-function from tool_choice if exists (reuse Llama 3.1 logic)
        # This prefix is added *after* stripping think block, as it forces the *content*
        llm_output = (
                self.get_generation_prefix_for_tool_choice(tool_choice) + llm_output
        )

        tool_calls = []
        text_response = ""

        func_prefix = "<function="
        end_func = "</function>"
        python_tag = "<|python_tag|>"  # Assuming code interpreter uses this tag

        # Reuse the parsing loop from Llama31Template, now operating on the
        # llm_output string *after* the think block has been removed.
        current_pos = 0
        while current_pos < len(llm_output):
            remaining_output = llm_output[current_pos:]

            # Check for code interpreter first (if applicable)
            if remaining_output.startswith(python_tag):
                # Consume any preceding text as part of the text response
                # (This case might be less common if function calls are expected immediately)
                if current_pos > 0 and text_response == "":
                    text_response = llm_output[:current_pos].strip()

                code = remaining_output[len(python_tag):]
                function_call = {
                    "name": "python",
                    "arguments": code,  # Arguments are the raw code string
                }
                tool_calls.append(
                    {
                        "type": "function",
                        "id": prompt_utils.get_random_tool_call_id(),
                        "function": function_call,
                    }
                )
                current_pos = len(llm_output)  # Consume the rest

            # Check for standard function call
            elif remaining_output.startswith(func_prefix):
                # Consume any preceding text
                if current_pos > 0 and text_response == "":
                    text_response = llm_output[:current_pos].strip()

                end_index = remaining_output.find(end_func)
                if end_index != -1:
                    # Extract the full call block <function=name>{args}</function>
                    full_call_block = remaining_output[:end_index + len(end_func)]
                    # Extract content between <function= and > for name, and after > for args
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
                        # Handle parsing error? Add to text_response? Log warning?
                        print(f"WARN: Could not parse function call content: {call_content}")
                        text_response += full_call_block  # Treat as text if parse fails

                    current_pos += len(full_call_block)  # Move past the function call
                else:
                    # Found <function= but no closing tag, treat rest as text
                    print(f"WARN: Found '{func_prefix}' but no closing '{end_func}'. Treating rest as text.")
                    text_response += remaining_output
                    current_pos = len(llm_output)
            else:
                # No function call detected at this position, advance character by character
                # Accumulate text response only at the very end or before a function call
                current_pos += 1

        # If after checking everything, no tool calls were made and we advanced, the whole thing (post-think) is text
        if not tool_calls and current_pos == len(llm_output) and text_response == "":
            text_response = llm_output.strip()

        # Clean up response structure
        if not text_response or text_response.strip() == "":
            text_response = None
        if not tool_calls:
            tool_calls = None

        # If only tool calls, ensure text response is None
        if tool_calls and text_response is not None and text_response.strip() == "":
            text_response = None

        # If we got both text and tool calls, it implies the text came *before* the first call
        # (Our loop logic assigns preceding text). Check if this is valid for the model.
        if text_response is not None and tool_calls is not None:
            print(
                f"WARN: Parsed both text content and tool calls. Text: '{text_response[:50]}...', Tool Calls: {len(tool_calls)}")

        final_content = None
        if think_content:
            final_content = think_content
            if text_response and text_response.strip():
                # Decide on separator, e.g., newline or space
                final_content += "\n" + text_response.strip()
        # Priority 2: Use text_response if no think block was captured
        elif text_response and text_response.strip():
            final_content = text_response.strip()
        return {"role": "assistant", "content": final_content, "tool_calls": tool_calls}

    # --- Streaming Logic Modifications ---

    def initialize_fsm_gen_state(
            self,
            tool_choice: Union[str, Tool],
            curr_text: str,
            curr_tokens: Optional[List[int]],
            add_code_interpreter: Optional[bool],
    ) -> Dict:
        """Initializes the FSM state, adding tracking for the think block."""
        # Initialize using Llama 3.1's logic first
        base_gen_state = super(Llama31Template, self).initialize_fsm_gen_state(  # Use Llama31 logic here
            tool_choice=tool_choice,
            curr_text=curr_text,
            curr_tokens=curr_tokens,
            add_code_interpreter=add_code_interpreter,
        )

        # Add state variables specific to handling the think block
        base_gen_state["is_in_think_block"] = True  # Start assuming we are in the think block
        base_gen_state["think_buffer"] = ""  # Buffer to hold content while in think block
        base_gen_state["processed_after_think"] = False  # Flag to re-run FSM logic once think block ends

        # If forcing a function call, we expect `<function=...>` *after* `<think>...</think>`
        # The initial stage from Llama31 logic (e.g., 'parameter') might be premature.
        # Let's reset the stage to 'pre-function' initially if 'is_in_think_block' is True.
        # The FSM update logic will transition correctly *after* the think block.
        # However, keep the forced 'func_name' if provided by tool_choice.
        if base_gen_state["is_in_think_block"]:
            # If tool_choice forced a function, stage might be 'parameter'.
            # Correct stage will be determined after '</think>' is processed.
            # For now, maybe 'pre-function' is safer, FSM update will fix it.
            # Let's keep the original stage, update_fsm will handle the post-think transition.
            pass  # Keep stage from Llama31 init for now.

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
        # --- MODIFICATION START: Handle finish reason within think block ---
        if finish_reason is not None:
            # If we finish while still theoretically in the think block (e.g., model error/max tokens)
            # or if the final state is parameter/code-interpreter (implying function call)
            final_finish_reason = finish_reason
            if gen_state["stage"] in ["parameter", "code-interpreter"] and not gen_state["is_in_think_block"]:
                final_finish_reason = "tool_calls"
            elif gen_state["is_in_think_block"]:
                print("WARN: Finish reason received while FSM state indicates still in think block.")
                # Decide on appropriate finish reason - maybe default to 'stop'?
                # Or trust the stage determined *if* processed_after_think is true?
                if gen_state["processed_after_think"] and gen_state["stage"] in ["parameter", "code-interpreter"]:
                    final_finish_reason = "tool_calls"
                else:
                    # Default to stop if we never got past think or ended in text-gen/pre-function
                    final_finish_reason = "stop"  # or keep original finish_reason? Let's keep original for now.

            # Return final (empty) delta based on determined reason
            return gen_state, prompt_utils.get_text_delta_response(
                None, False, final_finish_reason
            )
        # --- MODIFICATION END ---

        # If we are still in the think block according to the FSM state,
        # we don't yield any delta text yet. We just update the state.
        if gen_state["is_in_think_block"]:
            print(f"DEBUG: In think block, skipping delta generation for: '{delta_text}'")
            gen_state = self.update_fsm_gen_state(gen_state, delta_text, None, [], None)
            # If the update just finished the think block, we might want to yield
            # the initial delta for the *actual* content immediately.
            # Let's check if we just exited the think block AND buffer has content.
            if not gen_state["is_in_think_block"] and gen_state["processed_after_think"]:
                # Re-simulate the processing of the text *after* </think>
                # This is complex. Let's rely on the *next* delta_text call to yield correctly.
                # For now, return no delta.
                return gen_state, []  # Return empty list, not None
            else:
                return gen_state, []  # Return empty list, not None

        # If we are *not* in the think block, proceed with Llama 3.1's delta generation logic.
        # Make sure to pass the correct state and options.
        options = self.get_options_from_gen_state(
            gen_state=gen_state, tools_or_functions=tools_or_functions
        )

        # --- Reuse Llama31Template.stream_delta_text logic ---
        # (Copied and adapted slightly for context, ensuring correct state passing)
        responses = []
        if gen_state["stage"] == "text-gen":
            # Existing Llama31 logic for text-gen deltas
            if gen_state["first_chunk"]:
                responses.append(
                    prompt_utils.get_text_delta_response("", True, finish_reason)
                )
                # First chunk flag should only be set false *after* exiting think block
                # This happens implicitly as this code path is only run post-think.
                gen_state["first_chunk"] = False
            # ... (rest of Llama31 text-gen delta logic) ...
            # This part needs careful checking against original Llama31 logic for text buffering/handling special tokens
            # Simplified version: just pass delta through if not handling special cases here
            responses.append(
                prompt_utils.get_text_delta_response(
                    delta_text, False, finish_reason
                )
            )

        elif gen_state["stage"] == "parameter":
            # Existing Llama31 logic for parameter deltas
            if gen_state["first_function_chunk"]:
                responses.append(
                    prompt_utils.get_function_delta_response(
                        gen_state, "", True, gen_state["first_chunk"], finish_reason
                    )
                )
                gen_state["first_chunk"] = False  # Overall first chunk
                gen_state["first_function_chunk"] = False  # First chunk for *this* function
                # If initial state had text (e.g. from forced function call), send it now
                if gen_state["curr_text"] != "":
                    responses.append(
                        prompt_utils.get_function_delta_response(
                            gen_state, gen_state["curr_text"], False, False, finish_reason
                        )
                    )
            # ... (rest of Llama31 parameter delta logic, checking for </function>) ...
            # Simplified: pass delta through if not hitting end tags
            if "</" not in delta_text and "</" not in gen_state["curr_text"]:
                responses.append(
                    prompt_utils.get_function_delta_response(
                        gen_state, delta_text, False, False, finish_reason
                    )
                )
            elif "</" in delta_text:  # Handle partial end tag if needed
                suffix_match = re.search(r"</[^>]*$", delta_text)
                prefix = delta_text[:suffix_match.start()] if suffix_match else delta_text
                if prefix:
                    responses.append(
                        prompt_utils.get_function_delta_response(
                            gen_state, prefix, False, False, finish_reason
                        )
                    )
            # Note: Original Llama31 logic here is more robust for partial end tags.

        elif gen_state["stage"] == "code-interpreter":
            # Existing Llama31 logic for code-interpreter deltas
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
            # Usually transient, wait for parameter stage? Or handle name here?
            # Llama31 logic might buffer here. Let's assume no delta during 'function' stage.
            pass
        elif gen_state["stage"] == "pre-function":
            # Also transient, wait for text-gen or function.
            # Llama31 logic might buffer text here. Let's assume no delta.
            pass

        # Update state *after* determining deltas based on *previous* state
        gen_state = self.update_fsm_gen_state(
            gen_state=gen_state,
            new_token=delta_text,
            new_token_id=None,  # Assuming delta_text is string
            options=options,
            tokenizer=None,  # Tokenizer not needed if working with delta_text
        )

        return gen_state, responses

    def update_fsm_gen_state(
            self,
            gen_state: Dict,
            new_token: Optional[str],  # This is actually delta_text in streaming
            new_token_id: Optional[int],
            options: Optional[List],
            tokenizer: Any,  # Tokenizer might be None in simple streaming
    ) -> Dict:
        """Updates the FSM state, handling the transition out of the think block."""

        current_full_text = gen_state["curr_text"] + (new_token if new_token else "")
        gen_state["curr_text"] = current_full_text  # Update accumulated text

        # If we are currently marked as being in the think block:
        if gen_state["is_in_think_block"]:
            gen_state["think_buffer"] += (new_token if new_token else "")
            # Check if the think block has just ended
            if "</think>" in gen_state["think_buffer"]:
                print(f"DEBUG: Detected '</think>' in buffer: '{gen_state['think_buffer'][-20:]}'")
                gen_state["is_in_think_block"] = False
                # Extract content *after* the first </think>
                # Use regex to handle potential variations like </think >
                think_end_match = re.search(r"</think.*?>", gen_state["think_buffer"], re.IGNORECASE)
                if think_end_match:
                    post_think_text = gen_state["think_buffer"][think_end_match.end():]
                    gen_state["curr_text"] = post_think_text.lstrip()  # Reset curr_text to post-think content
                    print(f"DEBUG: Exited think block. Reset curr_text to: '{gen_state['curr_text'][:100]}...'")

                    # Clear think buffer and mark that we need to re-process the state
                    gen_state["think_buffer"] = ""
                    gen_state["processed_after_think"] = True  # Flag to indicate we need state transition

                    # --- Force state transition based on the new post-think curr_text ---
                    # This replicates the logic from the 'else' block below, but only runs once
                    # immediately after exiting the think block.

                    # Reset flags that might be relevant for the *new* content start
                    gen_state["first_chunk"] = True  # It's the first chunk of the *actual* response
                    gen_state["first_function_chunk"] = True

                    # Determine the new stage based on the post-think content
                    if gen_state["curr_text"].startswith("<|python_tag|>"):
                        gen_state = self._update_gen_state_for_fn_call(gen_state, "python")
                        gen_state["stage"] = "code-interpreter"
                        gen_state = self._reset_fsm_curr_text_and_tokens(
                            gen_state)  # Reset text after tag? Check Llama31
                    elif gen_state["curr_text"].startswith("<function="):
                        gen_state["stage"] = "function"
                        # Check if full function name is already present
                        match = re.match(r"<function=([^>]+)>", gen_state["curr_text"])
                        if match:
                            func_name = match.group(1)
                            gen_state = self._update_gen_state_for_fn_call(gen_state, func_name)
                            gen_state["stage"] = "parameter"
                            # Adjust curr_text to be only the arguments part
                            gen_state["curr_text"] = gen_state["curr_text"][match.end():]
                    elif gen_state["curr_text"].startswith("<"):
                        # Could be start of <function or <|python_tag|>, stay in pre-function
                        gen_state["stage"] = "pre-function"
                    elif len(gen_state["curr_text"]) > 0:
                        # It's starting with text
                        gen_state["stage"] = "text-gen"
                    else:
                        # Empty post-think text, stay in pre-function?
                        gen_state["stage"] = "pre-function"

                    print(f"DEBUG: Post-think state transition complete. New stage: {gen_state['stage']}")
                    # End of immediate post-think processing
                else:
                    # This shouldn't happen if </think> was detected, but log if it does
                    print("WARN: Detected '</think>' but regex failed to find end tag.")

            # If still in think block (or just processed the end tag), return updated state
            return gen_state

        # --- If not in think block, apply standard Llama 3.1 FSM logic ---
        # (Copied and adapted from Llama31Template.update_fsm_gen_state)
        # Note: This assumes curr_tokens is not used/needed if tokenizer=None
        stage = gen_state["stage"]
        curr_text = gen_state["curr_text"]

        if stage == "pre-function":
            # Add to text buffer only useful for Llama31's delta logic? Assume not needed here.
            # gen_state["text_buffer"].append(new_token)

            if curr_text.endswith("<|python_tag|>"):  # Check endswith for partial matches
                gen_state = self._update_gen_state_for_fn_call(gen_state, "python")
                gen_state["stage"] = "code-interpreter"
                gen_state = self._reset_fsm_curr_text_and_tokens(gen_state)  # Llama31 resets text here
            elif curr_text.endswith("<function="):
                gen_state["stage"] = "function"
            elif not curr_text.startswith("<"):  # If it started with text or became text
                gen_state["stage"] = "text-gen"

        elif stage == "text-gen":
            # Check if a function call starts mid-text
            if "<|python_tag|>" in curr_text:
                # Simplified: Transition immediately. Robust handling might need buffering.
                gen_state = self._update_gen_state_for_fn_call(gen_state, "python")
                gen_state["stage"] = "code-interpreter"
                gen_state = self._reset_fsm_curr_text_and_tokens(gen_state)
            elif "<function=" in curr_text:
                # Simplified: Transition immediately.
                gen_state["stage"] = "function"
                # Find where the call starts and potentially reset curr_text
                func_start_index = curr_text.find("<function=")
                gen_state["curr_text"] = curr_text[func_start_index:]  # Keep only from <function= onwards

        elif stage == "function":
            # Looking for the closing '>' of the function name
            pattern = r"<function=([^>]+)>"
            match = re.search(pattern, curr_text)
            if match:
                func_name = match.group(1)
                gen_state = self._update_gen_state_for_fn_call(gen_state, func_name)
                gen_state["stage"] = "parameter"
                # Update curr_text to contain only the part *after* <function=name>
                gen_state["curr_text"] = curr_text[match.end():]
                gen_state["first_function_chunk"] = True  # Reset for parameter delta

        elif stage == "parameter":
            # Looking for the end of the function call
            if "</function>" in curr_text:
                gen_state["stage"] = "pre-function"  # Ready for next text or function call
                # Update curr_text to be the part *after* </function>
                end_tag_index = curr_text.rfind("</function>")
                gen_state["curr_text"] = curr_text[end_tag_index + len("</function>"):]
                # Reset flags for next potential content
                gen_state["first_chunk"] = True
                gen_state["first_function_chunk"] = True

        # code-interpreter stage doesn't transition elsewhere in Llama31 logic easily,
        # assuming it consumes till stop token.

        return gen_state

    # Helper methods reused or adapted from Llama31Template/PromptTemplate base

    def _update_gen_state_for_fn_call(self, gen_state: Dict, func_name: str) -> Dict:
        """Updates gen_state when a function call is identified."""
        gen_state["func_name"] = func_name
        gen_state["func_index"] += 1
        gen_state["call_id"] = f"{prompt_utils.TOOL_CALL_ID_PREFIX}-{gen_state['func_index']}"
        # Reset flags specific to function content generation
        gen_state["first_function_chunk"] = True
        return gen_state

    def _reset_fsm_curr_text_and_tokens(self, gen_state: Dict) -> Dict:
        """Resets current text and tokens in FSM state."""
        gen_state["curr_text"] = ""
        gen_state["curr_tokens"] = []  # Assuming tokens might be used elsewhere
        return gen_state

    def get_options_from_gen_state(self, gen_state: Dict, tools_or_functions: List):
        # This function is related to constrained decoding grammar, which might
        # not be directly applicable/needed unless using specific inference engines.
        # Reusing Llama 3.1's empty implementation for now.
        return []

    def get_chat_template_jinja(self):
        return super().get_chat_template_jinja()

    def get_force_function_call_prefix(self, function_name: str):
        # The prefix forces the *content* after the think block
        return f"<function={function_name}>"

    def get_force_text_generation_prefix(self):
        # No prefix needed to force text (it's the default after think)
        return f""

    def get_tool_choice_required_prefix(self):
        # Forces *some* function call after the think block
        return "<function="

    def get_generation_prefix_for_tool_choice(self, tool_choice: Any) -> str:
        """Gets the prefix based on tool_choice constraint for generation"""
        # Reusing the logic helper from Llama31Template if possible, or reimplementing:
        prefix = ""
        if tool_choice == "none":
            prefix = self.get_force_text_generation_prefix()
        elif tool_choice == "required" or tool_choice == "auto":
            # For required/auto, the prefix depends on whether the FSM decides text or function first.
            # The most direct forcing is for a *specific* function or generally *a* function.
            # 'required' is handled by the <function= prefix generally.
            # Let's return empty here and let FSM handle 'auto'.
            # Return required prefix only for explicit 'required'.
            if tool_choice == "required":
                prefix = self.get_tool_choice_required_prefix()
        elif isinstance(tool_choice, (Tool, Function)) or (isinstance(tool_choice, dict) and "function" in tool_choice):
            # Assuming tool_choice can be Tool object or dict {type:"function", function:{name:"..."}}
            func_name = tool_choice.function.name if hasattr(tool_choice, 'function') else tool_choice.get('function',
                                                                                                           {}).get(
                'name')
            if func_name:
                prefix = self.get_force_function_call_prefix(func_name)
        return prefix
