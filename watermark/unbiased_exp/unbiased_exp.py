
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

# ============================================
# exp.py
# Description: Implementation of EXP algorithm
# ============================================

import torch
import scipy
import hashlib
import random
import torch.nn.functional as F
from typing import Tuple, Union
from functools import partial
from math import log, sqrt
from ..base import BaseWatermark
from utils.utils import load_config_file
from transformers import LogitsProcessor, LogitsProcessorList
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from visualize.data_for_visualization import DataForVisualization


class UnbiasedEXPConfig:
    """Config class for EXP algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/UnbiasedEXP.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'UnbiasedEXP':
            raise AlgorithmNameMismatchError('UnbiasedEXP', config_dict['algorithm_name'])

        random.seed(config_dict['unbiased_key'])
        hash_key = random.getrandbits(1024).to_bytes(128, "big")
        self.unbiased_hash_key = hash_key

        self.unbiased_gamma = config_dict['unbiased_gamma']
        self.unbiased_alpha = 0.5
        self.unbiased_ignore_history_generation = bool(config_dict['unbiased_ignore_history_generation'])
        self.unbiased_ignore_history_detection = bool(config_dict['unbiased_ignore_history_detection'])
        self.unbiased_z_threshold = config_dict['unbiased_z_threshold']
        self.unbiased_prefix_length = config_dict['unbiased_prefix_length']

        self.exp_prefix_length = config_dict['exp_prefix_length']
        self.exp_hash_key = config_dict['exp_hash_key']
        self.exp_threshold = config_dict['exp_threshold']
        self.exp_sequence_length = config_dict['exp_sequence_length']
        self.exp_top_k = config_dict['exp_top_k']

        self.token_entropy_threshold = config_dict['token_entropy_threshold']
        self.semantic_entropy_threshold = config_dict['semantic_entropy_threshold']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class UnbiasedEXPUtils:
    """Utility class for EXP algorithm, contains helper functions."""

    def __init__(self, config: UnbiasedEXPConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP utility class.

            Parameters:
                config (EXPConfig): Configuration for the EXP algorithm.
        """
        self.config = config
        self.exp_rng = torch.Generator()

        self.unbiased_rng = torch.Generator(device=self.config.device)
        self.cc_history = set()
        self.state_indicator = 0  # 0 for generation, 1 for detection and visualization

    def _get_rng_seed(self, context_code: any) -> int:
        """Get the random seed from the given context code and private key."""
        if (
                (not self.config.unbiased_ignore_history_generation and self.state_indicator == 0) or
                (not self.config.unbiased_ignore_history_detection and self.state_indicator == 1)
        ):
            self.cc_history.add(context_code)

        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.config.unbiased_hash_key)

        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2 ** 32 - 1)
        return seed

    def _extract_context_code(self, context: torch.LongTensor) -> bytes:
        """Extract context code from the given context."""
        if self.config.unbiased_prefix_length == 0:
            return context.detach().cpu().numpy().tobytes()
        else:
            return context[-self.config.unbiased_prefix_length:].detach().cpu().numpy().tobytes()

    def from_random(self, rng: Union[torch.Generator, list[torch.Generator]], vocab_size: int) -> torch.LongTensor:
        """Generate a permutation from the random number generator."""
        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack(
                [
                    torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
        return shuffle

    def reweight_logits(self, shuffle: torch.LongTensor, p_logits: torch.FloatTensor) -> torch.FloatTensor:
        """Reweight the logits using the shuffle and alpha."""
        unshuffle = torch.argsort(shuffle, dim=-1)

        s_p_logits = torch.gather(p_logits, -1, shuffle)
        s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)

        # normalize the log_cumsum to force the last element to be 0
        s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
        s_cumsum = torch.exp(s_log_cumsum)
        s_p = F.softmax(s_p_logits, dim=-1)

        boundary_1 = torch.argmax((s_cumsum > self.config.unbiased_alpha).to(torch.int), dim=-1, keepdim=True)
        p_boundary_1 = torch.gather(s_p, -1, boundary_1)
        portion_in_right_1 = (torch.gather(s_cumsum, -1, boundary_1) - self.config.unbiased_alpha) / p_boundary_1
        portion_in_right_1 = torch.clamp(portion_in_right_1, 0, 1)
        s_all_portion_in_right_1 = (s_cumsum > self.config.unbiased_alpha).type_as(p_logits)
        s_all_portion_in_right_1.scatter_(-1, boundary_1, portion_in_right_1)

        boundary_2 = torch.argmax((s_cumsum > (1 - self.config.unbiased_alpha)).to(torch.int), dim=-1, keepdim=True)
        p_boundary_2 = torch.gather(s_p, -1, boundary_2)
        portion_in_right_2 = (torch.gather(s_cumsum, -1, boundary_2) - (1 - self.config.unbiased_alpha)) / p_boundary_2
        portion_in_right_2 = torch.clamp(portion_in_right_2, 0, 1)
        s_all_portion_in_right_2 = (s_cumsum > (1 - self.config.unbiased_alpha)).type_as(p_logits)
        s_all_portion_in_right_2.scatter_(-1, boundary_2, portion_in_right_2)

        s_all_portion_in_right = s_all_portion_in_right_2 / 2 + s_all_portion_in_right_1 / 2
        s_shift_logits = torch.log(s_all_portion_in_right)
        shift_logits = torch.gather(s_shift_logits, -1, unshuffle)

        return p_logits + shift_logits

    def get_seed_for_cipher(self, input_ids: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Get the mask and seeds for the cipher."""
        batch_size = input_ids.size(0)
        context_codes = [
            self._extract_context_code(input_ids[i]) for i in range(batch_size)
        ]

        mask, seeds = zip(
            *[
                (context_code in self.cc_history, self._get_rng_seed(context_code))
                for context_code in context_codes
            ]
        )

        return mask, seeds

    def _get_green_token_quantile(self, input_ids: torch.LongTensor, vocab_size, current_token):
        """Get the vocab quantile of current token"""
        mask, seeds = self.get_seed_for_cipher(input_ids.unsqueeze(0))

        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]

        mask = torch.tensor(mask, device=input_ids.device)
        shuffle = self.from_random(
            rng, vocab_size
        )

        token_quantile = [(torch.where(shuffle[0] == current_token)[0] + 1) / vocab_size]
        return token_quantile, mask

    def _get_score(self, input_ids: torch.LongTensor, vocab_size):
        """Get the score of the input_ids"""
        scores = torch.zeros(input_ids.shape, device=input_ids.device)

        for i in range(input_ids.shape[-1] - 1):
            pre = input_ids[: i + 1]
            cur = input_ids[i + 1]
            token_quantile, mask = self._get_green_token_quantile(pre, vocab_size, cur)
            # if the current token is in the history and ignore_history_detection is False, set the score to -1
            if not self.config.unbiased_ignore_history_detection and mask[0]:
                scores[i + 1] = -1
            else:
                scores[i + 1] = torch.stack(token_quantile).reshape(-1)

        return scores

    def score_sequence(self, input_ids: torch.LongTensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        score = self._get_score(input_ids, self.config.vocab_size)
        green_tokens = torch.sum(score >= self.config.unbiased_gamma, dim=-1, keepdim=False)

        green_token_flags = torch.zeros_like(score)
        condition_indices = torch.nonzero(score >= self.config.unbiased_gamma, as_tuple=False)
        green_token_flags[condition_indices] = 1
        green_token_flags[:self.config.unbiased_prefix_length] = -1

        # Use two different ways to calculate z_score depending on whether to ignore history
        if not self.config.unbiased_ignore_history_detection:
            ignored_indices = torch.nonzero(score == -1, as_tuple=False)

            # Visualize the ignored tokens as ignored
            green_token_flags[ignored_indices] = -1

            # Calculate z_score using the sequence length after ignoring the ignored tokens
            sequence_length_for_calculation = input_ids.size(-1) - ignored_indices.size(0)
            z_score = (green_tokens - (1 - self.config.unbiased_gamma) * sequence_length_for_calculation) / sqrt(
                sequence_length_for_calculation)
        else:
            z_score = (green_tokens - (1 - self.config.unbiased_gamma) * input_ids.size(-1)) / sqrt(input_ids.size(-1))

        return z_score.item(), green_token_flags.tolist()

    # EXP
    def seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the random number generator with the last `prefix_length` tokens of the input."""
        time_result = 1
        for i in range(0, self.config.exp_prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.exp_rng.manual_seed(self.config.exp_hash_key * prev_token)
        return

    def exp_sampling(self, probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Sample a token from the vocabulary using the exponential sampling method."""

        # If top_k is not specified, use argmax
        if self.config.exp_top_k <= 0:
            return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)

        # Ensure top_k is not greater than the vocabulary size
        top_k = min(self.config.exp_top_k, probs.size(-1))

        # Get the top_k probabilities and their indices
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        # Perform exponential sampling on the top_k probabilities
        sampled_indices = torch.argmax(u.gather(-1, top_indices) ** (1 / top_probs), dim=-1)

        # Map back the sampled indices to the original vocabulary indices
        return top_indices.gather(-1, sampled_indices.unsqueeze(-1))

    def _value_transformation(self, value):
        """Transform the value to a range between 0 and 1."""
        return value / (value + 1)


class UnbiasedLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for DiP algorithm, process logits to add watermark."""

    def __init__(self, config: UnbiasedEXPConfig, utils: UnbiasedEXPUtils, *args, **kwargs) -> None:
        """
            Initialize the Unbiased logits processor.

            Parameters:
                config (UnbiasedConfig): Configuration for the DiP algorithm.
                utils (UnbiasedUtils): Utility class for the DiP algorithm.
        """
        self.config = config
        self.utils = utils

    def _apply_watermark(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> Tuple[
        torch.FloatTensor, torch.FloatTensor]:
        """Apply watermark to the scores."""
        mask, seeds = self.utils.get_seed_for_cipher(input_ids)

        rng = [
            torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds
        ]
        mask = torch.tensor(mask, device=scores.device)
        shuffle = self.utils.from_random(
            rng, scores.size(1)
        )

        reweighted_scores = self.utils.reweight_logits(shuffle, scores)

        return mask, reweighted_scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.unbiased_prefix_length:
            return scores

        mask, reweighted_scores = self._apply_watermark(input_ids, scores)

        if self.config.unbiased_ignore_history_generation:
            return reweighted_scores
        else:
            return torch.where(mask[:, None], scores, reweighted_scores)


class UnbiasedEXP(BaseWatermark):
    """Top-level class for the EXP algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = UnbiasedEXPConfig(algorithm_config, transformers_config)
        self.utils = UnbiasedEXPUtils(self.config)
        self.logits_processor = UnbiasedLogitsProcessor(self.config, self.utils)


    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text using the EXP algorithm."""

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
        for i in range(self.config.exp_sequence_length):
            with torch.no_grad():
                if past:
                    output = self.config.generation_model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.config.generation_model(inputs)

            scores = output.logits[:, -1, :self.config.vocab_size]

            if args[0] == 'S':
                if inputs.shape[-1] < self.config.unbiased_prefix_length:
                    return scores

                mask, reweighted_scores = self.logits_processor._apply_watermark(inputs, scores)

                # Get probabilities
                probs = torch.nn.functional.softmax(reweighted_scores, dim=-1).cpu()

                # Generate r1, r2,..., rk
                self.utils.seed_rng(inputs[0])
                random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.exp_rng)

                # Sample token to add watermark
                token = self.utils.exp_sampling(probs, random_numbers).to(self.config.device)

            elif args[0] == 'P':
                if i % 2 == 0:
                    # Set the state indicator to 0 for generation
                    self.utils.state_indicator = 0

                    # Configure generate_with_watermark
                    generate_with_watermark = partial(
                        self.config.generation_model.generate,
                        logits_processor=LogitsProcessorList([self.logits_processor]),
                        **self.config.gen_kwargs
                    )

                    # Generate watermarked text
                    token = generate_with_watermark(**encoded_prompt)[:, -1:]
                else:
                    # Get probabilities
                    probs = torch.nn.functional.softmax(scores, dim=-1).cpu()

                    # Generate r1, r2,..., rk
                    self.utils.seed_rng(inputs[0])
                    random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.exp_rng)

                    # Sample token to add watermark
                    token = self.utils.exp_sampling(probs, random_numbers).to(self.config.device)

            elif args[0] == 'H':
                result = args[2].get_entropy(inputs)
                token_entropy, semantic_entropy = result['token_entropy'], result['semantic_entropy']

                # print(f"token_entropy:{token_entropy}, semantic_entropy:{semantic_entropy}")

                if token_entropy > self.config.token_entropy_threshold and semantic_entropy > self.config.semantic_entropy_threshold:
                    A += 1
                    if inputs.shape[-1] < self.config.unbiased_prefix_length:
                        return scores

                    mask, reweighted_scores = self.logits_processor._apply_watermark(inputs, scores)

                    # Get probabilities
                    probs = torch.nn.functional.softmax(reweighted_scores, dim=-1).cpu()

                    # Generate r1, r2,..., rk
                    self.utils.seed_rng(inputs[0])
                    random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.exp_rng)

                    # Sample token to add watermark
                    token = self.utils.exp_sampling(probs, random_numbers).to(self.config.device)

                elif token_entropy < self.config.token_entropy_threshold and semantic_entropy < self.config.semantic_entropy_threshold:
                    B += 1
                    # Get probabilities
                    probs = torch.nn.functional.softmax(scores, dim=-1).cpu()

                    # Generate r1, r2,..., rk
                    self.utils.seed_rng(inputs[0])
                    random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.exp_rng)

                    # Sample token to add watermark
                    token = self.utils.exp_sampling(probs, random_numbers).to(self.config.device)

                elif token_entropy > self.config.token_entropy_threshold and semantic_entropy < self.config.semantic_entropy_threshold:
                    C += 1
                    # Set the state indicator to 0 for generation
                    self.utils.state_indicator = 0

                    # Configure generate_with_watermark
                    generate_with_watermark = partial(
                        self.config.generation_model.generate,
                        logits_processor=LogitsProcessorList([self.logits_processor]),
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


            # Update inputs
            inputs = torch.cat([inputs, token], dim=-1)

            # Update past
            past = output.past_key_values

            # Update attention mask
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        watermarked_tokens = inputs[0].detach().cpu()
        watermarked_text = self.config.generation_tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

        self.config.gen_kwargs['max_new_tokens'] = args[1]
        self.config.gen_kwargs['min_length'] = args[1] + 30

        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> dict:
        """Detect watermark in the text."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        unbaised_encoded_text = encoded_text.to(self.config.device)

        # Set the state indicator to 1 for detection
        self.utils.state_indicator = 1

        # Compute z-score using a utility method
        unbiased_z_score, _ = self.utils.score_sequence(unbaised_encoded_text)

        # Determine if the z-score indicates a watermark
        is_unbiased_watermarked = unbiased_z_score > self.config.unbiased_z_threshold

        # Clear the history
        self.utils.cc_history.clear()

        exp_encoded_text = encoded_text

        # Calculate the number of tokens to score, excluding the prefix
        num_scored = len(exp_encoded_text) - self.config.exp_prefix_length
        total_score = 0

        for i in range(self.config.exp_prefix_length, len(exp_encoded_text)):
            # Seed RNG with the prefix of the encoded text
            self.utils.seed_rng(exp_encoded_text[:i])

            # Generate random numbers for each token in the vocabulary
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.exp_rng)

            # Calculate score for the current token
            r = random_numbers[exp_encoded_text[i]]
            total_score += log(1 / (1 - r))

        # Calculate p_value
        exp_p_value = scipy.stats.gamma.sf(total_score, num_scored, loc=0, scale=1)

        # Determine if the computed score exceeds the threshold for watermarking
        is_exp_watermarked = exp_p_value < self.config.exp_threshold

        is_watermarked = bool(is_unbiased_watermarked | is_exp_watermarked)

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": {'Unbiased': unbiased_z_score, 'EXP': exp_p_value}}
        else:
            return (is_watermarked, [unbiased_z_score, exp_p_value])

    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        """Get data for visualization."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = \
        self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).numpy()[0]

        # Initialize the list of values with None for the prefix length
        highlight_values = [None] * self.config.exp_prefix_length

        # Calculate the value for each token beyond the prefix
        for i in range(self.config.exp_prefix_length, len(encoded_text)):
            # Seed the random number generator using the prefix of the encoded text
            self.utils.seed_rng(encoded_text[:i])
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.rng)
            r = random_numbers[encoded_text[i]]
            v = log(1 / (1 - r))
            v = self.utils._value_transformation(v)
            highlight_values.append(v)

        # Decode each token id to its corresponding string token
        decoded_tokens = [self.config.generation_tokenizer.decode([token_id]) for token_id in encoded_text]

        return DataForVisualization(decoded_tokens, highlight_values)

