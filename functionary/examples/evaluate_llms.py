import json
import os
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from llama_cpp import Llama
from transformers import AutoTokenizer
from tqdm import tqdm
import difflib
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

from functionary.prompt_template import get_prompt_template_by_version, PromptTemplate

# Load a mid-weight embedding model
_semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Load a sentiment-analysis pipeline (binary: POSITIVE/NEGATIVE)
_sentiment_pipe = pipeline(
    task='sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    tokenizer='distilbert-base-uncased-finetuned-sst-2-english'
)


def evaluate_llm_response_similarity(
        text_a: str,
        text_b: str,
        alpha: float = 0.8,
        reward_text_format=0.2
) -> dict:
    """
    Compute semantic and sentiment similarity between two texts and return:
      - semantic_similarity: cosine between embeddings (−1…1)
      - sentiment_similarity: 1 − |s_a − s_b|/2  (maps to 0…1)
      - overall_similarity: alpha * semantic + (1−alpha) * sentiment
      - reward_text_format: additional reward due to the model outputting correct format (text)

    Args:
      text_a, text_b: the two LLM outputs to compare
      alpha: weight for semantic vs. sentiment in overall score (0≤alpha≤1)

    Returns:
      dict with keys 'semantic_similarity', 'sentiment_similarity', 'overall_similarity'
    """
    # 1) Semantic similarity
    emb_a = _semantic_model.encode(text_a, convert_to_numpy=True)
    emb_b = _semantic_model.encode(text_b, convert_to_numpy=True)
    cos_sim = float(np.dot(emb_a, emb_b) /
                    (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))

    # 2) Sentiment scoring: map POSITIVE→+score, NEGATIVE→−score
    def _sent_score(txt: str) -> float:
        res = _sentiment_pipe(txt)[0]
        sign = 1 if res['label'] == 'POSITIVE' else -1
        return sign * res['score']

    try:
        s_a = _sent_score(text_a)
        s_b = _sent_score(text_b)
    except:
        return 0

    # Normalize difference to [0,1]: identical sentiment →1, opposite →0
    sent_sim = 1.0 - (abs(s_a - s_b) / 2.0)

    # 3) Combined score
    overall = alpha * cos_sim + (1.0 - alpha) * sent_sim

    return round(overall, 2) + reward_text_format


PATH_TO_TRAINED_GGUF = "../out_gguf/out_my_t_5e-Q4_K_M.gguf"
VANILLA_MODEL_REPO = "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
VANILLA_MODEL_FILENAME = "*Q4_K_M.gguf"
VALIDATION_DATASET_PATH = "glaive_parsed_test.jsonl"
TOKENIZER_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PROMPT_TEMPLATE_VERSION = "v3-llama3.1-deepseek-r1-think"

N_CTX = 2048
N_GPU_LAYERS = -1

MAX_EVAL_EXAMPLES = 100

print("Loading models...")
llm_trained = Llama(model_path=PATH_TO_TRAINED_GGUF, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=False)
# llm_vanilla = llm_trained
llm_vanilla = Llama.from_pretrained(
    repo_id=VANILLA_MODEL_REPO,
    filename=VANILLA_MODEL_FILENAME,
    verbose=False,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS
)

print("Loading tokenizer and prompt template...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, legacy=True)
prompt_template = get_prompt_template_by_version(PROMPT_TEMPLATE_VERSION)

stop_token_ids = [
    tokenizer.encode(token)[-1]
    for token in prompt_template.get_stop_tokens_for_generation()
]


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
    except:
        pass

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

    #
    # if not cleaned_output:
    #     return "text", ""

    parsed_result = prompt_template.parse_assistant_response(llm_output=raw_output, tool_choice=tools)
    if isinstance(parsed_result.get("tool_calls"), list):
        calls = parsed_result.get("tool_calls")
        normalized_calls = []
        for call in calls:
            call = call["function"]
            args = call.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args.replace("\'", "\""))
                except:
                    args = {}
            normalized_calls.append({"name": call.get("name"), "arguments": args})
        return "tool_calls", normalized_calls
    else:
        cleaned_output = remove_think_block(parsed_result.get("content", ""))
        return "text", cleaned_output


def parse_ground_truth_turn(assistant_message: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
    if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
        normalized_calls = []
        for call in assistant_message["tool_calls"]:
            func = call.get("function", {})
            args_str = func.get("arguments", "{}")
            if isinstance(args_str, str):
                args = json.loads(args_str)
            else:
                args = args_str
            normalized_calls.append({"name": func.get("name"), "arguments": args})
        return "tool_calls", normalized_calls
    else:
        return "text", assistant_message.get("content")


def compare_tool_calls_soft(expected: List[Dict], generated: List[Dict]) -> float:
    if not isinstance(expected, list) or not isinstance(generated, list):
        return 0.0

    if len(expected) != len(generated):
        return 0.0

    if not expected:
        return 1.0

    def sort_key(call):
        args_tuple = tuple(sorted(call.get("arguments", {}).items()))
        return (call.get("name", ""), str(args_tuple))

    sorted_expected = sorted(expected, key=sort_key)
    sorted_generated = sorted(generated, key=sort_key)

    matches = 0
    for exp_call, gen_call in zip(sorted_expected, sorted_generated):
        if exp_call.get("name") != gen_call.get("name"):
            continue
        matches += 0.5

        exp_args = exp_call.get("arguments", {})
        gen_args = gen_call.get("arguments", {})

        if exp_args == gen_args:
            matches += 0.5

    return matches / len(expected)


print("Starting validation...")
dataset = load_dataset(VALIDATION_DATASET_PATH)
if not dataset:
    print("No data loaded. Exiting.")
    exit()

if MAX_EVAL_EXAMPLES and MAX_EVAL_EXAMPLES > 0 and MAX_EVAL_EXAMPLES < len(dataset):
    print(f"Limiting evaluation to first {MAX_EVAL_EXAMPLES} examples.")
    dataset = dataset[:MAX_EVAL_EXAMPLES]

results_trained = {"correct_score": 0.0, "total_assistant_turns": 0, "skipped_turns": 0}
results_vanilla = {"correct_score": 0.0, "total_assistant_turns": 0, "skipped_turns": 0}

for i, example in enumerate(tqdm(dataset, desc="Evaluating Dialogues")):
    messages = example.get("messages", [])
    tools = example.get("tools", [])
    current_context = []

    print(f"\n===== Dialogue {i} =====")

    for turn_index, message in enumerate(messages):
        current_context.append(message)  # Add current message (user or assistant) to context

        if message.get("role") == "assistant":
            results_trained["total_assistant_turns"] += 1
            results_vanilla["total_assistant_turns"] += 1

            context_for_inference = current_context[:-1]  # Use context *before* this assistant message

            gt_type, gt_data = parse_ground_truth_turn(message)

            if gt_type == "error":
                print(f"  Turn {turn_index}: Skipping (Ground Truth Parse Error: {gt_data})")
                results_trained["skipped_turns"] += 1
                results_vanilla["skipped_turns"] += 1
                continue  # Skip this turn evaluation

            print(f"--- Evaluating Turn {turn_index} (Assistant) ---")
            print(f"  Ground Truth:  Type={gt_type}, Data={json.dumps(gt_data, indent=2)}")

            # --- Trained Model Inference & Eval ---
            output_trained_raw = run_inference(llm_trained, context_for_inference, tools, tokenizer, prompt_template,
                                               stop_token_ids)
            trained_score = 0.0
            is_skipped_trained = False

            pred_type_trained, pred_data_trained = parse_model_output(output_trained_raw, prompt_template,
                                                                      tools=tools)
            if gt_type == pred_type_trained:
                if gt_type == "text":
                    trained_score = evaluate_llm_response_similarity(pred_data_trained, gt_data)
                elif gt_type == "tool_calls":
                    # 0.2 for calling a function when needed, regardless if it's correct or not
                    trained_score = 0.2 + compare_tool_calls_soft(gt_data, pred_data_trained)
                    # bound to 1.0
                trained_score = min(trained_score, 1.0)
            if not is_skipped_trained:
                results_trained["correct_score"] += trained_score
            print(
                f"\nTrained Model: Type={pred_type_trained}, "
                f"Data={json.dumps(pred_data_trained, indent=2)},"
                f" Score={trained_score:.2f}")
            # f" Raw output: {output_trained_raw}")

            # --- Vanilla Model Inference & Eval ---
            output_vanilla_raw = run_inference(llm_vanilla, context_for_inference, tools, tokenizer, prompt_template,
                                               stop_token_ids)
            vanilla_score = 0.0
            is_skipped_vanilla = False

            pred_type_vanilla, pred_data_vanilla = parse_model_output(output_vanilla_raw, prompt_template,
                                                                      tools=tools)
            if gt_type == pred_type_vanilla:
                if gt_type == "text":
                    trained_score = evaluate_llm_response_similarity(pred_data_vanilla, gt_data)
                elif gt_type == "tool_calls":
                    vanilla_score = 0.2 + compare_tool_calls_soft(gt_data, pred_data_vanilla)
                trained_score = min(trained_score, 1.0)
            if not is_skipped_vanilla:
                results_vanilla["correct_score"] += vanilla_score
            print(
                f"\nVanilla Model: Type={pred_type_vanilla}, "
                f"Data={json.dumps(pred_data_vanilla, indent=2)}, "
                f"Score={vanilla_score:.2f}, ")
            # f"Raw output: {output_vanilla_raw}")
            print("-" * (28 + len(str(turn_index))))

    print(f"===== End Dialogue {i} =====")

# --- Calculate and Print Final Results ---
evaluated_turns_trained = results_trained["total_assistant_turns"] - results_trained["skipped_turns"]
evaluated_turns_vanilla = results_vanilla["total_assistant_turns"] - results_vanilla["skipped_turns"]

acc_trained = (results_trained["correct_score"] / evaluated_turns_trained) if evaluated_turns_trained > 0 else 0
acc_vanilla = (results_vanilla["correct_score"] / evaluated_turns_vanilla) if evaluated_turns_vanilla > 0 else 0

print("\n--- Overall Validation Results ---")
print(f"Dataset: {VALIDATION_DATASET_PATH}")
print(f"Total Dialogues Evaluated: {len(dataset)}")
print(f"Total Assistant Turns Encountered: {results_trained['total_assistant_turns']}")
print(
    f"Total Assistant Turns Skipped (Context/Error/Parse): {results_trained['skipped_turns']}")  # Assuming skips are similar for both
print("-" * 25)
print(f"Trained Model ({os.path.basename(PATH_TO_TRAINED_GGUF)}):")
print(f"  - Average Accuracy per Turn (Soft): {acc_trained:.4f}")
print(f"  - Total Correct Score Sum: {results_trained['correct_score']:.2f}")
print(f"  - Evaluated Turns: {evaluated_turns_trained}")
print("-" * 25)
print(f"Vanilla Model ({VANILLA_MODEL_REPO} - {VANILLA_MODEL_FILENAME}):")
print(f"  - Average Accuracy per Turn (Soft): {acc_vanilla:.4f}")
print(f"  - Total Correct Score Sum: {results_vanilla['correct_score']:.2f}")
print(f"  - Evaluated Turns: {evaluated_turns_vanilla}")
print("-" * 25)
