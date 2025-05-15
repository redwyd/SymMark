import scipy
import torch
import hashlib
import numpy as np
from functools import partial
from math import log, sqrt
from typing import Union
from ..base import BaseWatermark
from utils.utils import load_config_file
from transformers import LogitsProcessor, LogitsProcessorList
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from visualize.data_for_visualization import DataForVisualization


class UniGumbelConfig:
    """Config class for EXPGumbel algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXPGumbel configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/UniGumbel.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'UniGumbel':
            raise AlgorithmNameMismatchError('UniGumbel', config_dict['algorithm_name'])

        # Unigram
        self.gamma = config_dict['gamma']
        self.delta = config_dict['delta']
        self.hash_key = config_dict['hash_key']
        self.z_threshold = config_dict['z_threshold']

        # EXPGumbel
        self.prefix_length = config_dict['prefix_length']
        self.eps = config_dict['eps']
        self.threshold = config_dict['threshold']
        self.sequence_length = config_dict['sequence_length']
        self.temperature = config_dict['temperature']
        self.seed = config_dict['seed']

        # KgwExp
        self.token_entropy_threshold = config_dict['token_entropy_threshold']
        self.semantic_entropy_threshold = config_dict['semantic_entropy_threshold']
        self.k_means_top_k = config_dict['k_means_top_k']
        self.k_means_n_clusters = config_dict['k_means_n_clusters']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class UniGumbelUtils:
    """Utility class for EXPGumbel algorithm, contains helper functions."""

    def __init__(self, config: UniGumbelConfig, *args, **kwargs) -> None:
        """
            Initialize the EXPGumbel utility class.

            Parameters:
                config (EXPGumbelConfig): Configuration for the EXPGumbel algorithm.
        """
        self.config = config
        # self.mask = np.array([True] * int(self.config.gamma * self.config.vocab_size) +
        #                      [False] * (self.config.vocab_size - int(self.config.gamma * self.config.vocab_size)))
        # self.rng = np.random.default_rng(self._hash_fn(self.config.hash_key))
        # self.rng.shuffle(self.mask)

        # watermark stealing
        self.init_mask = np.array([True] * int(self.config.gamma * self.config.vocab_size) +
                                  [False] * (self.config.vocab_size - int(self.config.gamma * self.config.vocab_size)))
        self.mask = self.init_mask.copy()
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(0 % (2**64 - 1))  # safeguard against overflow from long
        vocab_permutation = torch.randperm(
            self.config.generation_tokenizer.vocab_size, device=self.config.device, generator=self.rng
        ).cpu().numpy()

        for index, value in enumerate(vocab_permutation):
            self.mask[value] = self.init_mask[index]


        self.generator = torch.Generator().manual_seed(self.config.seed)
        self.uniform = torch.clamp(
            torch.rand((self.config.vocab_size * self.config.prefix_length, self.config.vocab_size),
                       generator=self.generator, dtype=torch.bfloat16), min=self.config.eps)
        self.gumbel = (-torch.log(torch.clamp(-torch.log(self.uniform), min=self.config.eps))).to(self.config.device)

    # Unigram
    @staticmethod
    def _hash_fn(x: int) -> int:
        """hash function to generate random seed, solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        # watermark stealing
        return x
        # origin version
        # return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')

    def _compute_z_score(self, observed_count: int, T: int) -> float:
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def score_sequence(self, input_ids: torch.Tensor) -> tuple:
        """Score the input_ids and return z_score and green_token_flags."""
        num_tokens_scored = len(input_ids)
        green_token_count = 0
        green_token_flags = []
        for idx in range(0, len(input_ids)):
            curr_token = input_ids[idx]
            if self.mask[curr_token] == True:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)

        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags

    # EXPGumbel
    def _value_transformation(self, value):
        """Transform the value to a range between 0 and 1."""
        return value / (value + 1)


class UnigramLogitsProcessor(LogitsProcessor):
    """Logits processor for Unigram algorithm."""

    def __init__(self, config: UniGumbelConfig, utils: UniGumbelUtils, *args, **kwargs):
        """
            Initialize the Unigram logits processor.

            Parameters:
                config (UnigramConfig): Configuration for the Unigram algorithm.
                utils (UnigramUtils): Utility class for the Unigram algorithm.
        """
        self.config = config
        self.utils = utils
        self.green_list_mask = torch.tensor(self.utils.mask, dtype=torch.float32)

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor,
                               greenlist_bias: float) -> torch.Tensor:
        """Bias the logits for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process the logits and add watermark."""
        greenlist_mask = torch.zeros_like(scores)
        for i in range(input_ids.shape[0]):
            greenlist_mask[i] = self.green_list_mask
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=greenlist_mask.bool(),
                                             greenlist_bias=self.config.delta)
        return scores


class EXPGumbelLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for EXPGumbel algorithm, process logits to add watermark."""

    def __init__(self, config: UniGumbelConfig, utils: UniGumbelUtils, *args, **kwargs) -> None:
        """
            Initialize the EXPGumbel logits processor.

            Parameters:
                config (EXPGumbelConfig): Configuration for the KGW algorithm.
                utils (EXPGumbelUtils): Utility class for the KGW algorithm.
        """
        self.config = config
        self.utils = utils

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores
        prev_token = torch.sum(input_ids[:, -self.config.prefix_length:], dim=-1)  # (batch_size,)
        gumbel = self.utils.gumbel[prev_token]  # (batch_size, vocab_size)
        return scores[..., :gumbel.shape[-1]] / self.config.temperature + gumbel


class UniGumbel(BaseWatermark):
    """Top-level class for the EXPGumbel algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXPGumbel algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = UniGumbelConfig(algorithm_config, transformers_config)
        self.utils = UniGumbelUtils(self.config)
        self.unigram_logits_processor = UnigramLogitsProcessor(self.config, self.utils)
        self.exp_gumbel_logits_processor = EXPGumbelLogitsProcessor(self.config, self.utils)

    def watermark_logits_argmax(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.LongTensor:
        """
        Applies watermarking to the last token's logits and returns the argmax for that token.
        Returns tensor of shape (batch,), where each element is the index of the selected token.
        """

        # Get the logits for the last token
        last_logits = logits[:, -1, :]  # (batch, vocab_size)

        # Get the argmax of the logits
        last_token = torch.argmax(last_logits, dim=-1).unsqueeze(-1)  # (batch,)
        return last_token

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text using the EXPGumbel algorithm."""

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)

        # Initialize
        inputs = encoded_prompt['input_ids']
        attn = torch.ones_like(inputs)
        past = None

        A = 0
        B = 0
        C = 0
        D = 0

        self.config.gen_kwargs['max_new_tokens'] = 1
        self.config.gen_kwargs['min_length'] = 1

        # Generate tokens
        for i in range(self.config.sequence_length):
            with torch.no_grad():
                if past:
                    output = self.config.generation_model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
                    # output_gumbel = self.exp_gumbel_logits_processor(input_ids=inputs, scores=output.logits)
                else:
                    output = self.config.generation_model(inputs)
                    # output_gumbel = self.exp_gumbel_logits_processor(input_ids=inputs, scores=output.logits)

            scores = output.logits[:, -1, :self.config.vocab_size]

            if args[0] == 'S':

                greenlist_mask = torch.zeros_like(scores)
                for i in range(inputs.shape[0]):
                    greenlist_mask[i] = self.unigram_logits_processor.green_list_mask

                scores = self.unigram_logits_processor._bias_greenlist_logits(
                    scores=scores, greenlist_mask=greenlist_mask.bool(), greenlist_bias=self.config.delta
                )

                output.logits[:, -1, :self.config.vocab_size] = scores

                output_gumbel = self.exp_gumbel_logits_processor(input_ids=inputs, scores=output.logits)

                # Sample token
                token = self.watermark_logits_argmax(inputs, output_gumbel)

            elif args[0] == 'P':
                if i % 2 == 0:
                    # Configure generate_with_watermark
                    generate_with_watermark = partial(
                        self.config.generation_model.generate,
                        logits_processor=LogitsProcessorList([self.unigram_logits_processor]),
                        **self.config.gen_kwargs
                    )

                    # Generate watermarked text
                    token = generate_with_watermark(**encoded_prompt)[:, -1:]
                else:
                    output_gumbel = self.exp_gumbel_logits_processor(input_ids=inputs, scores=output.logits)

                    # Sample token
                    token = self.watermark_logits_argmax(inputs, output_gumbel)

            elif args[0] == 'H':
                result = args[2].get_entropy(inputs)
                token_entropy, semantic_entropy = result['token_entropy'], result['semantic_entropy']
                # print(f"token_entropy:{token_entropy}, semantic_entropy:{semantic_entropy}")

                if token_entropy > self.config.token_entropy_threshold and semantic_entropy < self.config.semantic_entropy_threshold:

                    greenlist_mask = torch.zeros_like(scores)
                    for i in range(inputs.shape[0]):
                        greenlist_mask[i] = self.unigram_logits_processor.green_list_mask
                    scores = self.unigram_logits_processor._bias_greenlist_logits(
                        scores=scores, greenlist_mask=greenlist_mask.bool(), greenlist_bias=self.config.delta
                    )

                    output.logits[:, -1, :self.config.vocab_size] = scores

                    output_gumbel = self.exp_gumbel_logits_processor(input_ids=inputs, scores=output.logits)

                    # Sample token
                    token = self.watermark_logits_argmax(inputs, output_gumbel)

                elif token_entropy < self.config.token_entropy_threshold and semantic_entropy < self.config.semantic_entropy_threshold:
                    output_gumbel = self.exp_gumbel_logits_processor(input_ids=inputs, scores=output.logits)

                    # Sample token
                    token = self.watermark_logits_argmax(inputs, output_gumbel)

                elif token_entropy > self.config.token_entropy_threshold and semantic_entropy > self.config.semantic_entropy_threshold:
                    # Configure generate_with_watermark
                    generate_with_watermark = partial(
                        self.config.generation_model.generate,
                        logits_processor=LogitsProcessorList([self.unigram_logits_processor]),
                        **self.config.gen_kwargs
                    )

                    # Generate watermarked text
                    token = generate_with_watermark(**encoded_prompt)[:, -1:]

                else:
                    D += 1
                    generate_with_watermark = partial(
                        self.config.generation_model.generate,
                        **self.config.gen_kwargs
                    )

                    # Generate watermarked text
                    token = generate_with_watermark(**encoded_prompt)[:, -1:]

            # Update past
            past = output.past_key_values

            # Update inputs
            inputs = torch.cat([inputs, token], dim=-1)

            # Update attention mask
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

            encoded_prompt['input_ids'] = inputs
            encoded_prompt['attention_mask'] = attn

        watermarked_tokens = inputs[0].detach().cpu()
        watermarked_text = self.config.generation_tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

        self.config.gen_kwargs['max_new_tokens'] = args[1]
        self.config.gen_kwargs['min_length'] = args[1] + 30

        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> dict:
        """Detect watermark in the text."""

        # Encode the text into tokens using the configured tokenizer
        exp_gumbel_encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)[0]
        unigram_encoded_text = exp_gumbel_encoded_text.to(self.config.device)

        # compute z_score
        unigram_z_score, _ = self.utils.score_sequence(unigram_encoded_text)

        # Determine if the z_score indicates a watermark
        is_unigram_watermarked = unigram_z_score > self.config.z_threshold

        seq_len = len(exp_gumbel_encoded_text)
        score = 0
        for i in range(self.config.prefix_length, seq_len):
            prev_tokens_sum = torch.sum(exp_gumbel_encoded_text[i - self.config.prefix_length:i], dim=-1)
            token = exp_gumbel_encoded_text[i]
            u = self.utils.uniform[prev_tokens_sum, token]
            score += log(1 / (1 - u))

        exp_gumbel_p_value = scipy.stats.gamma.sf(score, seq_len - self.config.prefix_length, loc=0, scale=1)

        # Determine if the computed score exceeds the threshold for watermarking
        is_exp_gumbel_watermarked = bool(exp_gumbel_p_value < self.config.threshold)

        is_watermarked = bool(is_unigram_watermarked | is_exp_gumbel_watermarked)

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": {'Unigram': unigram_z_score, 'ExpGumbel': exp_gumbel_p_value}}
        else:
            return (is_watermarked, [unigram_z_score, exp_gumbel_p_value])

    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        """Get data for visualization."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)[0]

        # Initialize the list of values with None for the prefix length
        highlight_values = [None] * self.config.prefix_length

        # Calculate the value for each token beyond the prefix
        for i in range(self.config.prefix_length, len(encoded_text)):
            # Seed the random number generator using the prefix of the encoded text
            prev_tokens_sum = torch.sum(encoded_text[i - self.config.prefix_length:i], dim=-1)
            token = encoded_text[i]
            u = self.utils.uniform[prev_tokens_sum, token]
            score = log(1 / (1 - u))
            highlight_values.append(self.utils._value_transformation(score))

        # Decode each token id to its corresponding token
        decoded_tokens = [self.config.generation_tokenizer.decode([token_id]) for token_id in encoded_text]

        return DataForVisualization(decoded_tokens, highlight_values)
