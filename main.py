import os
import re
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import warnings

import numpy as np
from transformers import logging

logging.set_verbosity_error()
warnings.filterwarnings('ignore')

import argparse
import torch
import json
from tqdm import tqdm

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from utils.utils import sampling_json, load_config_file, scorer, qa_f1_score, rouge_score, code_sim_score
from utils.compute_entropy import ComputeEntropy
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertForMaskedLM, T5Tokenizer, T5ForConditionalGeneration
from evaluation.dataset import C4Dataset
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, \
    DetectionPipelineReturnType
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator, \
    FundamentalSuccessRateCalculator
from evaluation.tools.text_editor import TruncatePromptTextEditor, WordDeletion, TruncateLLaMA3TaskTextEditor, \
    SynonymSubstitution, ContextAwareSynonymSubstitution, BackTranslationTextEditor, GPTParaphraser, DipperParaphraser, CopyPasteTextEditor
from evaluation.examples.assess_quality import assess_quality
from utils.openai_utils import OpenAIAPI


class Watermarking(object):
    def __init__(self, watermark_algorithm, watermark_type, dataset_path, dataset_name, data_size, target_model_name,
                 mode, attack_method, target_model_path, input_json_filename, output_json_filename, seed):
        self.mode = mode
        self.seed = seed
        self.watermark_list = ['KGW', 'Unigram', 'SWEET', 'EWD', 'SIR', 'XSIR', 'DIP', 'UPV', 'EXP', 'EXPEdit',
                               'EXPGumbel', 'Unbiased', 'SynthID', 'ITSEdit', 'KgwExp', 'UniGumbel', 'UnbiasedEXP',
                               'KgwGumbel', 'UniEXP', 'UnbiasedExpGumbel']
        self.watermark_algorithm = watermark_algorithm
        self.watermark_type = watermark_type
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.data_size = data_size
        self.attack_method = attack_method
        self.target_model_name = target_model_name
        self.target_model_path = target_model_path
        self.output_json_filename = output_json_filename
        self.input_json_filename = input_json_filename

        config_dict = load_config_file('config/DT.json')
        self.max_new_tokens = config_dict[self.dataset_name + '_max_new_tokens']
        self.min_new_tokens = config_dict[self.dataset_name + '_min_new_tokens']

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if self.mode == 'test' and self.attack_method != 'None':
            self.target_model_path = '/data/wangyidan/model/opt-1.3b'

        self.model = AutoModelForCausalLM.from_pretrained(self.target_model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.target_model_path)

        if 'opt' in self.target_model_name:
            self.vocab_size = 50272
        elif 'gpt-j' in self.target_model_name:
            self.vocab_size = 50400
        elif 'llama3' in self.target_model_name:
            self.vocab_size = 128256
        else:
            self.vocab_size = self.tokenizer.vocab_size

        # Transformers config
        self.transformers_config = TransformersConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            # vocab_size=50272 if 'opt' in self.target_model_name else 50400,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            # min_length=self.max_new_tokens + 30,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            # no_repeat_ngram_size=4
        )

        if not os.path.exists(self.input_json_filename):
            sampling_json(
                num=self.data_size,
                seed=seed,
                origin_file=f'./data/{dataset_name}/{dataset_name}.jsonl',
                sample_file=self.input_json_filename
            )

        if self.watermark_type == 'H':
            self.entropy_model = ComputeEntropy(
                model=self.model,
                tokenizer=self.tokenizer,
                top_k=256,
                n_clusters=10
            )
        else:
            self.entropy_model = None

        self.dataset2metric = {
            "t1": qa_f1_score,
            "t2": rouge_score,
            "t3": code_sim_score,
            "t4": rouge_score
        }

    def generate_watermark(self):
        # Check algorithm name
        assert self.watermark_algorithm in self.watermark_list

        # Load watermark algorithm
        myWatermark = AutoWatermark.load(
            f'{self.watermark_algorithm}',
            algorithm_config=f'config/{self.watermark_algorithm}.json',
            transformers_config=self.transformers_config
        )

        with open(self.dataset_path, 'r') as f1, open(self.output_json_filename, 'w') as f2:
            lines = f1.readlines()
            total_A = 0
            total_B = 0
            total_C = 0
            total_D = 0
            for line in tqdm(lines):
                line = json.loads(line)
                prompt = line['prompt']

                # Generate text
                if self.watermark_algorithm == 'I':
                    watermarked_text = myWatermark.generate_watermarked_text(prompt)
                else:
                    watermarked_text = myWatermark.generate_watermarked_text(
                        prompt, self.watermark_type, self.max_new_tokens, self.entropy_model
                    )
                    # total_A += A
                    # total_B += B
                    # total_C += C
                    # total_D += D
                
                unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)

                line['watermark_text'] = watermarked_text
                line['unwatermark_text'] = unwatermarked_text

                if self.watermark_algorithm == 'ITSEdit':
                    line['watermark_text'] = prompt + line['watermark_text']
                    line['unwatermark_text'] = prompt + line['unwatermark_text']

                f2.write(json.dumps(line, ensure_ascii=False) + '\n')

            # config_dict = load_config_file(f'config/{self.watermark_algorithm}.json')
            # TE = config_dict['token_entropy_threshold']
            # SE = config_dict['semantic_entropy_threshold']

            # radar_path = f"./radar/{self.dataset_name}_{self.data_size}/seed_{self.seed}/{self.target_model_name}/{self.watermark_algorithm}-{self.watermark_type}"
            # if not os.path.exists(radar_path):
            #     os.makedirs(radar_path)
            # with open(f'{radar_path}/TE_{TE}-SE_{SE}.jsonl', 'w') as f:
            #     result = {
            #         'A': total_A / (len(lines) * 200),
            #         'B': total_B / (len(lines) * 200),
            #         'C': total_C / (len(lines) * 200),
            #         'D': total_D / (len(lines) * 200)
            #     }
            #     f.write(json.dumps(result) + '\n')

    def evaluate_watermark(self, attack_name, unwatermarked_text_source):
        my_dataset = C4Dataset(self.output_json_filename, max_samples=self.data_size)

        my_watermark = AutoWatermark.load(
            f'{self.watermark_algorithm}',
            algorithm_config=f'config/{self.watermark_algorithm}.json',
            transformers_config=self.transformers_config
        )

        attack_list = [TruncatePromptTextEditor() if self.dataset_name == 'c4' or self.dataset_name == 'opengen' else TruncateLLaMA3TaskTextEditor()]
        # attack_list = [TruncatePromptTextEditor()]
        # attack_list = []
        if attack_name == 'Word-D':
            attack_list.append(WordDeletion(ratio=0.3))
        elif attack_name == 'Word-S-DICT':
            attack_list.append(SynonymSubstitution(ratio=0.5))
        elif attack_name == 'Word-S-BERT':
            attack_list.append(ContextAwareSynonymSubstitution(
                ratio=0.5,
                tokenizer=BertTokenizer.from_pretrained('/data/wangyidan/model/bert-large-uncased/'),
                model=BertForMaskedLM.from_pretrained('/data/wangyidan/model/bert-large-uncased/').to(self.device)
            ))
        elif attack_name == 'Copy-Paste':
            attack_list.append(CopyPasteTextEditor(times=1))
        elif attack_name == 'Doc-P-GPT':
            # attack_list.append(GPTParaphraser(openai_model='gpt-4', prompt='Please rewrite the following text: '))
            attack_list.append(DipperParaphraser(
                tokenizer=T5Tokenizer.from_pretrained('/data/wangyidan/model/t5-v1_1-xxl/'),
                model=T5ForConditionalGeneration.from_pretrained('/data/wangyidan/model/dipper-paraphraser-xxl/',device_map='auto'),
                lex_diversity=80, order_diversity=80, sent_interval=1, max_new_tokens=200, do_sample=True, top_p=0.75, top_k=None
            ))
        elif attack_name == 'Translation':
            # attack_list.append(GPTParaphraser(openai_model='gpt-3.5-turbo', prompt='Please translate the following text from English to Chinese, then back to English, and finally back to English only: '))
            attack_list.append(BackTranslationTextEditor(self.device))
        elif 'Doc-P-Dipper' in attack_name:
            attack_list.append(DipperParaphraser(
                tokenizer=T5Tokenizer.from_pretrained('/data/wangyidan/model/t5-v1_1-xxl/'),
                model=T5ForConditionalGeneration.from_pretrained('/data/wangyidan/model/dipper-paraphraser-xxl/',device_map='auto'),
                lex_diversity=60, order_diversity=0, sent_interval=1, max_new_tokens=200, do_sample=True, top_p=0.75, top_k=None
            ))
        elif 'Doc-P-Dipper-1' in attack_name:
            attack_list.append(DipperParaphraser(
                tokenizer=T5Tokenizer.from_pretrained('/data/wangyidan/model/t5-v1_1-xxl/'),
                model=T5ForConditionalGeneration.from_pretrained('/data/wangyidan/model/dipper-paraphraser-xxl/',device_map='auto'),
                lex_diversity=60, order_diversity=60, sent_interval=1, max_new_tokens=200, do_sample=True, top_p=0.75, top_k=None
            ))
        else:
            attack_name = None
            print(f'attack_name:{attack_name}')

        pipeline1 = WatermarkedTextDetectionPipeline(
            dataset=my_dataset,
            text_editor_list=attack_list,
            show_progress=True,
            return_type=DetectionPipelineReturnType.SCORES,
            # return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        )

        pipeline2 = UnWatermarkedTextDetectionPipeline(
            dataset=my_dataset,
            text_editor_list=attack_list,
            show_progress=True,
            text_source_mode=unwatermarked_text_source,
            return_type=DetectionPipelineReturnType.SCORES,
            # return_type=DetectionPipelineReturnType.IS_WATERMARKED,
        )

        if attack_name and self.watermark_type == 'I':
            labels = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC', 'FPRs', 'TPRs', 'AUROC']
        elif attack_name and self.watermark_type != 'I':
            labels = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC', 'FPRs', 'TPRs', 'AUROC', 'Thresholds']
        else:
            labels = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC', 'AUROC']

        calculator1 = DynamicThresholdSuccessRateCalculator(
            labels=labels,
            rule='best',
            reverse=True if 'EXP' in self.watermark_algorithm or 'Gumbel' in self.watermark_algorithm or 'Edit' in self.watermark_algorithm else False
        )

        calculator2 = FundamentalSuccessRateCalculator(
            labels=labels,
        )

        calculator3 = DynamicThresholdSuccessRateCalculator(
            labels=labels,
            rule='target_fpr',
            target_fpr=0,
            reverse=True if 'EXP' in self.watermark_algorithm or 'Gumbel' in self.watermark_algorithm else False
        )
        #
        # metric_list = ['F1', 'ACC'] if 't' in self.dataset_name else ['F1', 'PPL']
        # if self.data_size == 50:
        #     metric_list = ['GPT4']
        # metric_list = ['F1'] if attack_name else metric_list
        # metric_list = ['ACC']
        metric_list = ['F1', 'PPL']
        for metric in metric_list:
            if metric == 'F1':
                if self.watermark_type == 'I':
                    detect_eval_result = calculator1.calculate(
                        pipeline1.evaluate(my_watermark),
                        pipeline2.evaluate(my_watermark),
                    )
                else:
                    detect_eval_result = calculator1.calculate(
                        pipeline1.evaluate(my_watermark, self.watermark_type),
                        pipeline2.evaluate(my_watermark, self.watermark_type),
                        watermark_type=self.watermark_type
                    )
                print(f"\033[31m{self.watermark_algorithm}({self.watermark_type}): {detect_eval_result}\033[0m")

                if attack_name:
                    auroc_path = f"./auroc/{self.dataset_name}_{self.data_size}/seed_{self.seed}/{self.target_model_name}/{self.watermark_algorithm}-{self.watermark_type}"
                    if not os.path.exists(auroc_path):
                        os.makedirs(auroc_path)
                    with open(f'{auroc_path}/{attack_name}.jsonl', 'w') as f:
                        f.write(json.dumps(detect_eval_result) + '\n')

                        # if self.watermark_type == 'I':
                        #     f.write(json.dumps(detect_eval_result) + '\n')
                        # else:
                        #     FPRs = [0, detect_eval_result['FPR'], 1]
                        #     TPRs = [0, detect_eval_result['TPR'], 1]
                        #     detect_eval_result['FPRs'] = np.linspace(0, 1, 10)
                        #     detect_eval_result['TPRs'] = np.interp(detect_eval_result['FPRs'], FPRs, TPRs)
                        #     detect_eval_result['FPRs'] = detect_eval_result['FPRs'].tolist()
                        #     detect_eval_result['TPRs'] = detect_eval_result['TPRs'].tolist()
                        #     f.write(json.dumps(detect_eval_result) + '\n')

            if metric == 'PPL':
                ppl_eval_result = assess_quality(
                    algorithm_name=self.watermark_algorithm,
                    metric=metric,
                    transformers_config=self.transformers_config,
                    eval_file=self.output_json_filename,
                    unwatermarked_text_source=unwatermarked_text_source
                )

                ppl_mean_score = {
                    'watermarked': sum([result['watermarked'] for result in ppl_eval_result]) / len(ppl_eval_result),
                    'unwatermarked': sum([result['unwatermarked'] for result in ppl_eval_result]) / len(ppl_eval_result)
                }

                print(f"\033[32m{metric}:{ppl_mean_score}\033[0m")


                if unwatermarked_text_source == 'natural':
                    output_path = './ppl/{self.dataset_name}_{self.data_size}/seed_{self.seed}/'
                    os.makedirs(output_path)

                    with open(output_path + 'natural.jsonl', 'w') as f:
                        natural_text_ppl_list = []
                        for res in ppl_eval_result:
                            natural_text_ppl_list.append(res['unwatermarked'])
                        f.write(json.dumps({'natural': natural_text_ppl_list}))
                else:
                    output_path = './ppl/{self.dataset_name}_{self.data_size}/seed_{self.seed}/{self.target_model_name}/'
                    os.makedirs(output_path)

                    with open(output_path + '{self.watermark_algorithm}-{self.watermark_type}.jsonl', 'w') as f:
                        watermark_text_ppl_list = []
                        unwatermark_text_ppl_list = []
                        for res in ppl_eval_result:
                            watermark_text_ppl_list.append(res['watermarked'])
                            unwatermark_text_ppl_list.append(res['unwatermarked'])
                        f.write(json.dumps({'watermarked': watermark_text_ppl_list, 'unwatermarked': unwatermark_text_ppl_list}))

            if metric == 'ACC':
                with open(self.output_json_filename, 'r') as f:
                    lines = f.readlines()
                    watermark_predictions = []
                    unwatermarked_predictions = []
                    ground_truths = []
                    for line in lines:
                        line = json.loads(line)

                        watermarked_prediction = line['watermark_text'][line['prompt_length']:]
                        # print(watermarked_prediction)

                        unwatermarked_prediction = line['unwatermark_text'][line['prompt_length']:] if unwatermarked_text_source == 'generated' else line['natural_text'][line['prompt_length']:]
                        # print(unwatermarked_prediction)

                        watermark_predictions.append(watermarked_prediction)
                        unwatermarked_predictions.append(unwatermarked_prediction)
                        ground_truths.append(line['natural_text'])

                    watermark_score = scorer(self.dataset2metric[self.dataset_name], watermark_predictions, ground_truths)
                    unwatermark_score = scorer(self.dataset2metric[self.dataset_name], unwatermarked_predictions, ground_truths)
                    print(f'watermark_score: {watermark_score}, unwatermark_score: {unwatermark_score}')

            if metric == 'GPT4':
                detect_eval_result = calculator1.calculate(
                    pipeline1.evaluate(my_watermark, self.watermark_type),
                    pipeline2.evaluate(my_watermark, self.watermark_type),
                    watermark_type=self.watermark_type
                )

                client = OpenAIAPI(model="gpt-4o", temperature=0.0, system_content="You are given a prompt and a response, and you need grade the response out of 100 based on: Accuracy (20 points) - correctness and relevance to the prompt; Detail (20 points) - comprehensiveness and depth; Grammar and Typing (30 points) - grammatical and typographical accuracy; Vocabulary (30 points) - appropriateness and richness. Deduct points for shortcomings in each category. Note that you only need to give an overall score, no explanation is required.")
                scores = []
                with open(self.output_json_filename, 'r') as f:
                    lines = f.readlines()
                    prompts = []
                    watermark_predictions = []
                    for line in tqdm(lines):
                        line = json.loads(line)
                        prompts.append(line['prompt'])
                        watermark_predictions.append(line['watermark_text'][len(line['prompt']):])

                        response = client.get_result_from_gpt3_5(f'prompt: {prompts[0]}\n response: {watermark_predictions[0]}')
                        pattern = r'-?\d+\.?\d*'
                        scores.append(int(re.search(pattern, response.choices[0].message.content).group()) / 100)
                print(f'GPT4 Score: {sum(scores) / len(scores)}')

                config_dict = load_config_file(f'config/{self.watermark_algorithm}.json')
                TE = config_dict['token_entropy_threshold']
                SE = config_dict['semantic_entropy_threshold']

                radar_path = f"./radar/{self.dataset_name}_{self.data_size}/seed_{self.seed}/{self.target_model_name}/{self.watermark_algorithm}-{self.watermark_type}"
                with open(f'{radar_path}/TE_{TE}-SE_{SE}.jsonl', 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = json.loads(line)
                        line['F1'] = detect_eval_result['F1']
                        line['TPR'] = detect_eval_result['TPR']
                        line['TNR'] = detect_eval_result['TNR']
                        line['GPT4'] = sum(scores) / len(scores)

                        detect_eval_result.update(line)

                with open(f'{radar_path}/TE_{TE}-SE_{SE}.jsonl', 'w') as f:
                    f.write(json.dumps(detect_eval_result) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='training or evaluation')
    parser.add_argument('--seed', type=str, default='random seed')
    parser.add_argument('--attack_method', type=str, default='different attack methods for texts')
    parser.add_argument('--text_source', type=str, default='natural texts or generated texts')
    parser.add_argument('--dataset_path', type=str, help='the path of the dataset')
    parser.add_argument('--dataset_name', type=str, help='the name of the dataset')
    parser.add_argument('--dataset_size', type=int, help='the size of the dataset')
    parser.add_argument('--watermark_algorithm', type=str, help='the name of the watermark algorithm')
    parser.add_argument('--watermark_type', type=str, help='the type of the watermark algorithm')
    parser.add_argument('--target_model_name', type=str, help='the name of the target model')
    parser.add_argument('--target_model_path', type=str, help='the path of the target model')
    parser.add_argument('--input_json_filename', type=str, help='the filename of the input json file')
    parser.add_argument('--output_json_filename', type=str, help='the filename of the output json file')
    args = parser.parse_args()

    print(args)

    watermarking = Watermarking(
        seed=args.seed,
        mode=args.mode,
        watermark_algorithm=args.watermark_algorithm,
        watermark_type=args.watermark_type,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        data_size=args.dataset_size,
        attack_method=args.attack_method,
        target_model_name=args.target_model_name,
        target_model_path=args.target_model_path,
        input_json_filename=args.input_json_filename,
        output_json_filename=args.output_json_filename,
    )

    if args.mode == 'train':
        watermarking.generate_watermark()
    elif args.mode == 'test':
        watermarking.evaluate_watermark(attack_name=args.attack_method, unwatermarked_text_source=args.text_source)
