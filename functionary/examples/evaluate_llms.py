import json
import os
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from llama_cpp import Llama
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
import difflib
from sentence_transformers import SentenceTransformer
import numpy as np

from functionary.prompt_template import get_prompt_template_by_version, PromptTemplate, \
    get_prompt_template_from_tokenizer

_semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
_sentiment_pipe = pipeline(
    task='sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    tokenizer='distilbert-base-uncased-finetuned-sst-2-english'
)

# --- Configuration ---
MODEL_CONFIGS = [
    {
        "name": "trained",
        "model_loader": lambda: Llama(model_path="../out_gguf/out_my_t_5e-Q4_K_M.gguf", n_ctx=N_CTX,
                                      n_gpu_layers=N_GPU_LAYERS, verbose=False),
        "tokenizer_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "prompt_template_version": "v3-llama3.1-deepseek-r1-think",
        "has_think_block": True
    },
    {
        "name": "vanilla",
        "model_loader": lambda: Llama.from_pretrained(repo_id="bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
                                                      filename="*Q4_K_M.gguf", verbose=False, n_ctx=N_CTX,
                                                      n_gpu_layers=N_GPU_LAYERS),
        "tokenizer_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "prompt_template_version": "v3-llama3.1-deepseek-r1-think",
        "has_think_block": True
    },
    {
        "name": "meetkai",
        "model_loader": lambda: Llama.from_pretrained(repo_id="meetkai/functionary-small-v2.5-GGUF", filename="*Q4*",
                                                      verbose=False, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS),
        "tokenizer_name": "meetkai/functionary-small-v2.5-GGUF",  # Specific tokenizer for this model
        "prompt_template_version": None,  # Get template from tokenizer
        "has_think_block": False
    }
]

VALIDATION_DATASET_PATH = "glaive_parsed_test.jsonl"
N_CTX = 2048
N_GPU_LAYERS = -1
MAX_EVAL_EXAMPLES = 100  # Set to None or 0 to evaluate all examples


# --- Helper Functions ---

def evaluate_llm_response_similarity(
        text_a: str,
        text_b: str,
        alpha: float = 0.8,
        reward_text_format=0.2
) -> dict:
    emb_a = _semantic_model.encode(text_a, convert_to_numpy=True)
    emb_b = _semantic_model.encode(text_b, convert_to_numpy=True)
    cos_sim = float(np.dot(emb_a, emb_b) /
                    (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))

    def _sent_score(txt: str) -> float:
        res = _sentiment_pipe(txt)[0]
        sign = 1 if res['label'] == 'POSITIVE' else -1
        return sign * res['score']

    try:
        s_a = _sent_score(text_a)
        s_b = _sent_score(text_b)
    except:
        return 0

    sent_sim = 1.0 - (abs(s_a - s_b) / 2.0)
    overall = alpha * cos_sim + (1.0 - alpha) * sent_sim
    return round(overall, 2) + reward_text_format


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        print(f"Error: Dataset file not found at {filepath}")
        exit(1)
    dataset = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def run_inference(
        llm: Llama,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        prompt_template: PromptTemplate,
        stop_token_ids: List[int],
) -> str:
    inference_messages = messages + [{"role": "assistant"}]

    for i, message in enumerate(messages):
        if "tool_calls" in message and isinstance(message["tool_calls"], list):
            if isinstance(message["tool_calls"][0]["function"]["arguments"], dict):
                messages[i]["tool_calls"][0]["function"]["arguments"] = str(
                    message["tool_calls"][0]["function"]["arguments"])
    try:
        prompt_str = prompt_template.get_prompt_from_messages(inference_messages, tools)
    except:
        print("\nERROR", inference_messages, tools, "\n\n")
        return ""
    token_ids = tokenizer.encode(prompt_str)

    if len(token_ids) >= llm.n_ctx():
        return ""

    gen_tokens = []
    output_generator = llm.generate(token_ids, temp=0)
    try:
        for token_id in output_generator:
            if token_id in stop_token_ids:
                break
            gen_tokens.append(token_id)
    except Exception as e:
        # Only catch the exception during generation itself
        print(f"\nLLAMA.CPP GENERATION ERROR (Dialogue Skipped for this model): {e}")
        return ""

    if gen_tokens and gen_tokens[-1] in stop_token_ids:
        gen_tokens = gen_tokens[:-1]

    llm_output = tokenizer.decode(gen_tokens).strip()
    return llm_output


def remove_think_block(text: str) -> str:
    think_end_tag = "</think>"
    end_index = text.find(think_end_tag)
    if end_index != -1:
        return text[end_index + len(think_end_tag):].lstrip()
    return text


def parse_model_output(raw_output: str, prompt_template: PromptTemplate, tools: list) -> Tuple[str, Optional[Any]]:
    if not raw_output:
        return "error", "Empty output from model"

    parsed_result = prompt_template.parse_assistant_response(llm_output=raw_output, tool_choice="auto")

    if isinstance(parsed_result.get("tool_calls"), list):
        calls = parsed_result.get("tool_calls")
        normalized_calls = []
        for call in calls:
            # Handle potential nested structure if parse_assistant_response doesn't fully normalize
            func = call.get("function") if isinstance(call.get("function"), dict) else call
            args = func.get("arguments", {})
            if isinstance(args, str):
                args_str_cleaned = args.replace("\'", "\"").replace("True", "true").replace("False", "false").replace(
                    "None", "null")
                try:
                    args = json.loads(args_str_cleaned)
                except json.JSONDecodeError:
                    args = {"__parse_error__": args}  # Keep original string if parsing fails
            normalized_calls.append({"name": func.get("name"), "arguments": args})
        return "tool_calls", normalized_calls
    else:
        # Return content directly, thinking block should be removed before calling this
        return "text", parsed_result.get("content", "")


def parse_ground_truth_turn(assistant_message: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
    if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
        normalized_calls = []
        for call in assistant_message["tool_calls"]:
            func = call.get("function", {})
            args_str = func.get("arguments", "{}")
            if isinstance(args_str, str):
                args = json.loads(args_str)
            else:
                args = args_str  # Assume it's already a dict if not a string
            normalized_calls.append({"name": func.get("name"), "arguments": args})
        return "tool_calls", normalized_calls
    else:
        # Remove think block from ground truth text content as well
        gt_content = assistant_message.get("content", "")
        cleaned_gt_content = remove_think_block(gt_content)
        return "text", cleaned_gt_content


def compare_tool_calls_soft(expected: List[Dict], generated: List[Dict]) -> float:
    if not isinstance(expected, list) or not isinstance(generated, list):
        return 0.0

    if len(expected) != len(generated):
        return 0.0

    if not expected:
        return 1.0

    def sort_key(call):
        # Handle potential unhashable types like dicts in arguments
        args_repr = repr(tuple(sorted(call.get("arguments", {}).items())))
        return (call.get("name", ""), args_repr)

    sorted_expected = sorted(expected, key=sort_key)
    sorted_generated = sorted(generated, key=sort_key)

    matches = 0
    for exp_call, gen_call in zip(sorted_expected, sorted_generated):
        if exp_call.get("name") != gen_call.get("name"):
            continue
        matches += 0.5  # Credit for correct name

        exp_args = exp_call.get("arguments", {})
        gen_args = gen_call.get("arguments", {})

        # Check for parse errors before comparing directly
        if isinstance(exp_args, dict) and "__parse_error__" in exp_args: continue
        if isinstance(gen_args, dict) and "__parse_error__" in gen_args: continue

        if exp_args == gen_args:
            matches += 0.5  # Credit for matching arguments

    return matches / len(expected)


# --- Initialization ---
print("Loading models, tokenizers, and templates...")
loaded_models = {}
results = {}

for config in MODEL_CONFIGS:
    model_name = config["name"]
    print(f"  Loading {model_name}...")
    llm = config["model_loader"]()
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], legacy=True)

    if config["prompt_template_version"]:
        prompt_template = get_prompt_template_by_version(config["prompt_template_version"])
    else:
        prompt_template = get_prompt_template_from_tokenizer(tokenizer)

    stop_token_ids = [
        tokenizer.encode(token)[-1]
        for token in prompt_template.get_stop_tokens_for_generation()
    ]

    loaded_models[model_name] = {
        "llm": llm,
        "tokenizer": tokenizer,
        "prompt_template": prompt_template,
        "stop_token_ids": stop_token_ids,
        "has_think_block": config["has_think_block"]
    }
    results[model_name] = {"correct_score": 0.0, "total_assistant_turns": 0, "skipped_turns": 0}

# --- Main Evaluation ---
print("Starting validation...")
dataset = load_dataset(VALIDATION_DATASET_PATH)
if not dataset:
    print("No data loaded. Exiting.")
    exit()

if MAX_EVAL_EXAMPLES and MAX_EVAL_EXAMPLES > 0 and MAX_EVAL_EXAMPLES < len(dataset):
    print(f"Limiting evaluation to first {MAX_EVAL_EXAMPLES} examples.")
    dataset = dataset[:MAX_EVAL_EXAMPLES]

total_dialogues = len(dataset)

for i, example in enumerate(tqdm(dataset, desc="Evaluating Dialogues")):
    messages = example.get("messages", [])
    tools = example.get("tools", [])
    current_context = []

    # print(f"\n===== Dialogue {i} =====") # Reduce print frequency

    for turn_index, message in enumerate(messages):
        # IMPORTANT: Use a deep copy for context manipulation if needed,
        # but here we just append to a running list which is fine.
        current_context.append(message)

        if message.get("role") == "assistant":
            context_for_inference = current_context[:-1]
            gt_type, gt_data = parse_ground_truth_turn(message)

            if gt_type == "error":
                # print(f"  Turn {turn_index}: Skipping (Ground Truth Parse Error: {gt_data})")
                for model_name in loaded_models:
                    results[model_name]["skipped_turns"] += 1
                continue

            print(f"\n--- Evaluating Turn {turn_index} (Assistant) ---")
            print(f"  Ground Truth:  Type={gt_type}, Data={json.dumps(gt_data, indent=2)}")

            for model_name, model_data in loaded_models.items():
                results[model_name]["total_assistant_turns"] += 1
                score = 0.0
                is_skipped = False

                output_raw = run_inference(
                    model_data["llm"],
                    context_for_inference,
                    tools,
                    model_data["tokenizer"],
                    model_data["prompt_template"],
                    model_data["stop_token_ids"]
                )

                if not output_raw:
                    # Check for context length issue implicitly (empty output from run_inference)
                    prompt_str_check = model_data["prompt_template"].get_prompt_from_messages(
                        context_for_inference + [{"role": "assistant"}], tools)
                    token_ids_check = model_data["tokenizer"].encode(prompt_str_check)
                    if len(token_ids_check) >= model_data["llm"].n_ctx():
                        results[model_name]["skipped_turns"] += 1
                        is_skipped = True
                        pred_type, pred_data = "skipped", "Context Length Exceeded"
                    else:
                        pred_type, pred_data = "error", "Empty/Failed Inference"
                else:
                    # Process output: remove think block if necessary BEFORE parsing
                    cleaned_output = remove_think_block(output_raw) if model_data["has_think_block"] else output_raw
                    pred_type, pred_data = parse_model_output(cleaned_output, model_data["prompt_template"],
                                                              tools=tools)

                if not is_skipped and pred_type != "error":
                    if gt_type == pred_type:
                        if gt_type == "text":
                            score = evaluate_llm_response_similarity(pred_data, gt_data) if gt_data and pred_data else (
                                1.0 if not gt_data and not pred_data else 0.0)
                        elif gt_type == "tool_calls":
                            score = 0.2 + compare_tool_calls_soft(gt_data, pred_data)
                        score = min(score, 1.0)  # Ensure score doesn't exceed 1.0

                if not is_skipped:
                    results[model_name]["correct_score"] += score

                # Optional: Print turn-level comparison (can be verbose)
                print(f"  {model_name}: Type={pred_type}, Score={score:.2f}, Data={json.dumps(pred_data, indent=2)}")

            print("-" * (28 + len(str(turn_index))))

    print(f"===== End Dialogue {i} =====\n") # Reduce print frequency

# --- Calculate and Print Final Results ---
print("\n--- Overall Validation Results ---")
print(f"Dataset: {VALIDATION_DATASET_PATH}")
print(f"Total Dialogues Evaluated: {total_dialogues}")

total_assistant_turns = 0
total_skipped_turns = 0
# Use results from one model for total turns/skips, assuming they are processed consistently
if results:
    first_model_name = list(results.keys())[0]
    total_assistant_turns = results[first_model_name]["total_assistant_turns"]
    total_skipped_turns = results[first_model_name]["skipped_turns"]

print(f"Total Assistant Turns Encountered: {total_assistant_turns}")
print(f"Total Assistant Turns Skipped (Context/Error/Parse): {total_skipped_turns}")
print("-" * 25)

for model_name, res in results.items():
    evaluated_turns = res["total_assistant_turns"] - res["skipped_turns"]
    avg_acc = (res["correct_score"] / evaluated_turns) if evaluated_turns > 0 else 0
    model_id = f"{model_name} ({os.path.basename(loaded_models[model_name]['llm'].model_path)})" if hasattr(
        loaded_models[model_name]['llm'], 'model_path') else model_name

    print(f"{model_id}:")
    print(f"  - Average Accuracy per Turn (Soft): {avg_acc:.4f}")
    print(f"  - Total Correct Score Sum: {res['correct_score']:.2f}")
    print(f"  - Evaluated Turns: {evaluated_turns}")
    print("-" * 25)
