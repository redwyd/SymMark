# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================
# utils.py
# Description: Utility functions
# ==============================

import re
import os
import json
import random
import string

from tqdm import tqdm
from collections import Counter
from rouge import Rouge
from fuzzywuzzy import fuzz

from nltk.tokenize import word_tokenize


def load_config_file(path: str) -> dict:
    """Load a JSON configuration file from the specified path and return it as a dictionary."""
    try:
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return config_dict

    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{path}': {e}")
        # Handle other potential JSON decoding errors here
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Handle other unexpected errors here
        return None


def load_json_as_list(input_file: str) -> list:
    """Load a JSON file as a list of dictionaries."""
    res = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        d = json.loads(line)
        res.append(d)
    return res


def create_directory_for_file(file_path) -> None:
    """Create the directory for the specified file path if it does not already exist."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def sampling_json(origin_file, sample_file, num, seed) -> None:
    """Sample n instances from the specified dataset"""
    with open(origin_file, 'r') as f1, open(sample_file, 'w') as f2:
        random.seed(seed)
        lines = random.sample(f1.readlines(), num)
        print("Data Sampling ...")
        for line in tqdm(lines):
            line = json.loads(line)
            f2.write(json.dumps(line, ensure_ascii=False) + '\n')


def f1_score(prediction, ground_truth, **kwargs):
    # prediction = [pred.replace('assistant', '') for pred in prediction]
    prediction = ['true' if 'true' in pred.lower() else pred for pred in prediction]
    prediction = ['false' if 'false' in pred.lower() else pred for pred in prediction]
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    else:
        return num_same


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)


def scorer(eval_function, predictions, answers):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, eval_function(prediction, ground_truth))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == '__main__':
    # with open('../dataset/factual_knowledge/knowledge_understanding.jsonl', 'r') as f1, open('../dataset/t1/t1_100.jsonl', 'w') as f2:
    #     lines = f1.readlines()[:100]
    #     for line in tqdm(lines):
    #         line = json.loads(line)
    #         # t1
    #         prompt = f"[INST] Please give answer to the following question about knowledge. Note: If you are asked for true or false, just answer \"true\" or \"false\" only. If you are asked for similarity, just answer with the entity name only. Do not give anything other than the answers. Question:\n{line['context']}\n{line['input']}[/INST]"
    #
    #         t1_dict = {
    #             # 'prompt': ,
    #             'prompt': prompt,
    #             # 'prompt': f"You are a helpful assistant, please answer the following question within 200 words:\n{line['context']}\n{line['input']}",
    #             # 'prompt': f"Please complete the code given below. \n{line['context']}Next line of code:\n",
    #             # 'prompt': f"You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{line['context']}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    #             'natural_text': line['outputs'],
    #             # 'prompt_length': len(prompt) - len(
    #             #     "<|begin_of_text|><|start_header_id|>" + "<|end_header_id|>" + "<|eot_id|><|start_header_id|>" + "<|end_header_id|>")
    #             'prompt_length': len(prompt)
    #         }
    #         f2.write(json.dumps(t1_dict) + '\n')
    with open('../dataset/t5/tofel_real_91.json', 'r') as f1, open('../dataset/t5/t5_91.jsonl', 'w') as f2, open('../output/I/Unigram/c4/200/seed_42/opt-6.7b.jsonl') as f3:
        lines1 = json.load(f1)
        lines2 = f3.readlines()[:91]
        for line1, line2 in tqdm(zip(lines1, lines2)):
            line2 = json.loads(line2)
            f2_dict = {
                'prompt': "",
                'watermark_text': line2['watermark_text'],
                'unwatermark_text': line2['unwatermark_text'],
                'natural_text': line1['document']
            }
            f2.write(json.dumps(f2_dict, ensure_ascii=False) + '\n')