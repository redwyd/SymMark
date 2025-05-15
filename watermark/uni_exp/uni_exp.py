import torch
import scipy
import hashlib
import numpy as np
from functools import partial
from typing import Union
from math import log, sqrt
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.compute_entropy import ComputeEntropy
from transformers import LogitsProcessor, LogitsProcessorList
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from visualize.data_for_visualization import DataForVisualization


class UniEXPConfig:
    """Config class for EXP algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/UniEXP.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'UniEXP':
            raise AlgorithmNameMismatchError('UniEXP', config_dict['algorithm_name'])

        # Unigram
        self.gamma = config_dict['gamma']
        self.delta = config_dict['delta']
        self.unigram_hash_key = config_dict['unigram_hash_key']
        self.z_threshold = config_dict['z_threshold']

        # EXP
        self.prefix_length = config_dict['prefix_length']
        self.exp_hash_key = config_dict['exp_hash_key']
        self.threshold = config_dict['threshold']
        self.sequence_length = config_dict['sequence_length']
        self.top_k = config_dict['top_k']

        self.token_entropy_threshold = config_dict['token_entropy_threshold']
        self.semantic_entropy_threshold = config_dict['semantic_entropy_threshold']
        self.k_means_top_k = config_dict['k_means_top_k']
        self.k_means_n_clusters = config_dict['k_means_n_clusters']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class UniEXPUtils:
    """Utility class for EXP algorithm, contains helper functions."""

    def __init__(self, config: UniEXPConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP utility class.

            Parameters:
                config (EXPConfig): Configuration for the EXP algorithm.
        """
        self.config = config
        # origin version
        # self.mask = np.array([True] * int(self.config.gamma * self.config.vocab_size) +
        #                      [False] * (self.config.vocab_size - int(self.config.gamma * self.config.vocab_size)))
        # self.unigram_rng = np.random.default_rng(self._hash_fn(self.config.unigram_hash_key))
        # self.unigram_rng.shuffle(self.mask)

        # watermark stealing
        self.init_mask = np.array([True] * int(self.config.gamma * self.config.vocab_size) +
                                  [False] * (self.config.vocab_size - int(self.config.gamma * self.config.vocab_size)))
        self.mask = self.init_mask.copy()
        # self.rng = np.random.default_rng(self._hash_fn(self.config.hash_key))
        # self.rng.shuffle(self.mask)
        # print(self.mask)

        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(0 % (2**64 - 1))  # safeguard against overflow from long
        vocab_permutation = torch.randperm(
            self.config.generation_tokenizer.vocab_size, device=self.config.device, generator=self.rng
        ).cpu().numpy()

        for index, value in enumerate(vocab_permutation):
            self.mask[value] = self.init_mask[index]

        self.exp_rng = torch.Generator()

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

    def calculate_entropy(self, model, tokenized_text: torch.Tensor):
        """Calculate entropy for each token in the tokenized_text."""
        with torch.no_grad():
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            probs = torch.softmax(output.logits, dim=-1)

            top_k_probs, top_k_indices = torch.topk(probs, self.config.k_means_top_k, dim=-1)

            token_embeddings = self.config.generation_model.get_input_embeddings()(top_k_indices)
            token_embeddings = token_embeddings.cpu().clone().detach().numpy()

            entropy_model = ComputeEntropy(
                model=self.config.generation_model,
                tokenizer=self.config.generation_tokenizer,
                top_k=self.config.k_means_top_k,
                n_clusters=self.config.k_means_n_clusters
            )

            token_embeddings = token_embeddings.squeeze()
            top_k_probs = top_k_probs.squeeze()

            semantic_entropy_list = []
            for i in range(top_k_probs.size(0)):
                embeddings_i = token_embeddings[i]
                probs_i = top_k_probs[i].cpu().detach().numpy()
                semantic_entropy, _ = entropy_model.semantic_clustering(probs_i, embeddings_i)
                semantic_entropy_list.append(semantic_entropy)

            token_entropy_list = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
            token_entropy_list = token_entropy_list[0].cpu().tolist()
            token_entropy_list.insert(0, -10000.0)
            return token_entropy_list[:-1], semantic_entropy_list

    def unigram_origin_score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
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

    # Parallel
    def unigram_parallel_score_sequence(self, input_ids):
        """Score the input_ids based on 50/50."""
        num_tokens_scored = int(len(input_ids) / 2)

        green_token_count = 0
        green_token_flags = []

        for idx in range(0, len(input_ids)):
            if idx % 2 == 0:
                curr_token = input_ids[idx]
                if self.mask[curr_token] == True:
                    green_token_count += 1
                    green_token_flags.append(1)
                else:
                    green_token_flags.append(0)

        z_score_1 = self._compute_z_score(green_token_count, num_tokens_scored)

        green_token_count = 0
        green_token_flags = []

        for idx in range(0, len(input_ids)):
            if idx % 2 == 1:
                curr_token = input_ids[idx]
                if self.mask[curr_token] == True:
                    green_token_count += 1
                    green_token_flags.append(1)
                else:
                    green_token_flags.append(0)

        z_score_2 = self._compute_z_score(green_token_count, num_tokens_scored)

        return z_score_1 if z_score_1 > z_score_2 else z_score_2

    # Hybrid
    def unigram_entropy_score_sequence(self, input_ids, entropy_list):
        """Score the input_ids based on the greenlist and entropy."""
        num_tokens_scored = (len(input_ids) - len([e for e in entropy_list if e <= self.config.token_entropy_threshold]))

        green_token_flags = []
        weights = []

        for idx in range(0, len(input_ids)):
            curr_token = input_ids[idx]
            if self.mask[curr_token] == True:
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
            if entropy_list[idx] > self.config.token_entropy_threshold:
                weights.append(1)
            else:
                weights.append(0)

        # calculate number of green tokens where weight is 1
        green_token_count = sum([1 for i in range(len(green_token_flags)) if green_token_flags[i] == 1 and weights[i] == 1])
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)

        return z_score, green_token_flags, weights

    # EXP
    def seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the random number generator with the last `prefix_length` tokens of the input."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.exp_rng.manual_seed(self.config.exp_hash_key * prev_token)
        return

    def exp_sampling(self, probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Sample a token from the vocabulary using the exponential sampling method."""

        # If top_k is not specified, use argmax
        if self.config.top_k <= 0:
            return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)

        # Ensure top_k is not greater than the vocabulary size
        top_k = min(self.config.top_k, probs.size(-1))

        # Get the top_k probabilities and their indices
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        # Perform exponential sampling on the top_k probabilities
        sampled_indices = torch.argmax(u.gather(-1, top_indices) ** (1 / top_probs), dim=-1)

        # Map back the sampled indices to the original vocabulary indices
        return top_indices.gather(-1, sampled_indices.unsqueeze(-1))

    def _value_transformation(self, value):
        """Transform the value to a range between 0 and 1."""
        return value / (value + 1)

    def exp_origin_score_sequence(self, encoded_text):
        num_scored = len(encoded_text) - self.config.prefix_length
        total_score = 0

        for i in range(self.config.prefix_length, len(encoded_text)):
            # Seed RNG with the prefix of the encoded text
            self.seed_rng(encoded_text[:i])

            # Generate random numbers for each token in the vocabulary
            random_numbers = torch.rand(self.config.vocab_size, generator=self.exp_rng)

            # Calculate score for the current token
            r = random_numbers[encoded_text[i]]
            total_score += log(1 / (1 - r))

        # Calculate p_value
        p_value = scipy.stats.gamma.sf(total_score, num_scored, loc=0, scale=1)

        return p_value

    def exp_parallel_score_sequence(self, encoded_text):
        num_scored = (len(encoded_text) - self.config.prefix_length) / 2
        total_score = 0

        for i in range(self.config.prefix_length, len(encoded_text)):
            if i % 2 == 0:
                # Seed RNG with the prefix of the encoded text
                self.seed_rng(encoded_text[:i])

                # Generate random numbers for each token in the vocabulary
                random_numbers = torch.rand(self.config.vocab_size, generator=self.exp_rng)

                # Calculate score for the current token
                r = random_numbers[encoded_text[i]]
                total_score += log(1 / (1 - r))

        # Calculate p_value
        p_value_1 = scipy.stats.gamma.sf(total_score, num_scored, loc=0, scale=1)


        for i in range(self.config.prefix_length, len(encoded_text)):
            if i % 2 == 1:

                # Seed RNG with the prefix of the encoded text
                self.seed_rng(encoded_text[:i])

                # Generate random numbers for each token in the vocabulary
                random_numbers = torch.rand(self.config.vocab_size, generator=self.exp_rng)

                # Calculate score for the current token
                r = random_numbers[encoded_text[i]]
                total_score += log(1 / (1 - r))

        # Calculate p_value
        p_value_2 = scipy.stats.gamma.sf(total_score, num_scored, loc=0, scale=1)

        return p_value_1 if p_value_1 < p_value_2 else p_value_2

    def exp_entropy_score_sequence(self, input_ids, entropy_list):
        """Score the input_ids based on the greenlist and entropy."""
        num_tokens_scored = (len(input_ids) - self.config.prefix_length -
                             len([e for e in entropy_list[self.config.prefix_length:] if
                                  e >= self.config.semantic_entropy_threshold]))
        if num_tokens_scored < 1:
            # raise ValueError(
            #     (
            #         f"Must have at least {1} token to score after "
            #     )
            # )
            return 0

        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        weights = [-1 for _ in range(self.config.prefix_length)]

        total_score = 0

        for i in range(self.config.prefix_length, len(input_ids)):
            if entropy_list[i] > self.config.semantic_entropy_threshold:
                continue
            # Seed RNG with the prefix of the encoded text
            self.seed_rng(input_ids[:i])

            # Generate random numbers for each token in the vocabulary
            random_numbers = torch.rand(self.config.vocab_size, generator=self.exp_rng)

            # Calculate score for the current token
            r = random_numbers[input_ids[i]]
            total_score += log(1 / (1 - r))

        # Calculate p_value
        p_value = scipy.stats.gamma.sf(total_score, num_tokens_scored, loc=0, scale=1)

        return p_value


class UnigramLogitsProcessor(LogitsProcessor):
    """Logits processor for Unigram algorithm."""

    def __init__(self, config: UniEXPConfig, utils: UniEXPUtils, *args, **kwargs):
        """
            Initialize the Unigram logits processor.

            Parameters:
                config (UnigramConfig): Configuration for the Unigram algorithm.
                utils (UnigramUtils): Utility class for the Unigram algorithm.
        """
        self.config = config
        self.utils = utils
        self.green_list_mask = torch.tensor(self.utils.mask, dtype=torch.float32)

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the logits for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process the logits and add watermark."""
        greenlist_mask = torch.zeros_like(scores)
        for i in range(input_ids.shape[0]):
            greenlist_mask[i] = self.green_list_mask
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=greenlist_mask.bool(), greenlist_bias=self.config.delta)
        return scores


class UniEXP(BaseWatermark):
    """Top-level class for the EXP algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = UniEXPConfig(algorithm_config, transformers_config)
        self.utils = UniEXPUtils(self.config)
        self.logits_processor = UnigramLogitsProcessor(self.config, self.utils)


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

        token_total_entropy = 0
        semantic_total_entropy = 0

        self.config.gen_kwargs['max_new_tokens'] = 1
        self.config.gen_kwargs['min_new_tokens'] = 1

        # Generate tokens
        for i in range(self.config.sequence_length):
            with torch.no_grad():
                if past:
                    output = self.config.generation_model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.config.generation_model(inputs)

            scores = output.logits[:, -1, :self.config.vocab_size]

            if args[0] == 'S':
                greenlist_mask = torch.zeros_like(scores)
                for i in range(inputs.shape[0]):
                    greenlist_mask[i] = self.logits_processor.green_list_mask
                scores = self.logits_processor._bias_greenlist_logits(
                    scores=scores, greenlist_mask=greenlist_mask.bool(), greenlist_bias=self.config.delta
                )

                # Get probabilities
                probs = torch.nn.functional.softmax(scores, dim=-1).cpu()

                # Generate r1, r2,..., rk
                self.utils.seed_rng(inputs[0])
                random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.exp_rng)

                # Sample token to add watermark
                token = self.utils.exp_sampling(probs, random_numbers).to(self.config.device)

            elif args[0] == 'P':
                if i % 2 == 0:
                    # greenlist_mask = torch.zeros_like(scores)
                    # for i in range(inputs.shape[0]):
                    #     greenlist_mask[i] = self.logits_processor.green_list_mask
                    # scores = self.logits_processor._bias_greenlist_logits(
                    #     scores=scores, greenlist_mask=greenlist_mask.bool(), greenlist_bias=self.config.delta
                    # )
                    #
                    # # Get probabilities
                    # probs = torch.nn.functional.softmax(scores, dim=-1).cpu()
                    # token = torch.multinomial(probs, 1).to(self.config.device)
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
                result = args[2].get_entropy(inputs[:, -256:])
                token_entropy, semantic_entropy = result['token_entropy'], result['semantic_entropy']

                token_total_entropy += token_entropy
                semantic_total_entropy += semantic_entropy

                if token_entropy > self.config.token_entropy_threshold and semantic_entropy < self.config.semantic_entropy_threshold:
                    A += 1
                    greenlist_mask = torch.zeros_like(scores)
                    for i in range(inputs.shape[0]):
                        greenlist_mask[i] = self.logits_processor.green_list_mask
                    scores = self.logits_processor._bias_greenlist_logits(scores=scores,
                                                                                  greenlist_mask=greenlist_mask.bool(),
                                                                                  greenlist_bias=self.config.delta)

                    # Get probabilities
                    probs = torch.nn.functional.softmax(scores, dim=-1).cpu()

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

                elif token_entropy > self.config.token_entropy_threshold and semantic_entropy > self.config.semantic_entropy_threshold:
                    C += 1
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

            encoded_prompt['input_ids'] = inputs
            encoded_prompt['attention_mask'] = attn

        watermarked_tokens = inputs[0].detach().cpu()
        watermarked_text = self.config.generation_tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

        if args[0] == 'H':
            print(f"A: {A}, B: {B}, C: {C}, D: {D} ", end="")
            print(f"Mean Token Entropy: {token_total_entropy / self.config.sequence_length}, Mean Semantic Entropy: {semantic_total_entropy / self.config.sequence_length}", end="")

        self.config.gen_kwargs['max_new_tokens'] = args[1]
        self.config.gen_kwargs['min_new_tokens'] = args[1]

        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> dict:
        """Detect watermark in the text."""

        # Encode the text
        exp_encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        unigram_encoded_text = exp_encoded_text.to(self.config.device)

        if args[0] == 'S':
            # Compute z_score using a utility method
            z_score, _ = self.utils.unigram_origin_score_sequence(unigram_encoded_text)
            # Calculate the number of tokens to score, excluding the prefix
            p_value = self.utils.exp_origin_score_sequence(exp_encoded_text)

        elif args[0] == 'P':
            z_score = self.utils.unigram_parallel_score_sequence(unigram_encoded_text)
            p_value = self.utils.exp_parallel_score_sequence(exp_encoded_text)
            # z_score, _ = self.utils.unigram_origin_score_sequence(unigram_encoded_text)
            # p_value = self.utils.exp_origin_score_sequence(exp_encoded_text)

        else:
            # token_entropy_list, semantic_entropy_list = self.utils.calculate_entropy(model=self.config.generation_model, tokenized_text=unigram_encoded_text)
            # z_score, _, _ = self.utils.unigram_entropy_score_sequence(unigram_encoded_text, token_entropy_list)
            # p_value = self.utils.exp_entropy_score_sequence(exp_encoded_text, semantic_entropy_list)
            z_score, _ = self.utils.unigram_origin_score_sequence(unigram_encoded_text)
            p_value = self.utils.exp_origin_score_sequence(exp_encoded_text)


        # Determine if the z_score indicates a watermark
        is_kgw_watermarked = z_score > self.config.z_threshold

        # Determine if the computed score exceeds the threshold for watermarking
        is_exp_watermarked = p_value < self.config.threshold

        is_watermarked = bool(is_kgw_watermarked | is_exp_watermarked)

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": {'Unigram': z_score, 'EXP': p_value}}
        else:
            return (is_watermarked, [z_score, p_value])

    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        """Get data for visualization."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = \
        self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).numpy()[0]

        # Initialize the list of values with None for the prefix length
        highlight_values = [None] * self.config.prefix_length

        # Calculate the value for each token beyond the prefix
        for i in range(self.config.prefix_length, len(encoded_text)):
            # Seed the random number generator using the prefix of the encoded text
            self.utils.seed_rng(encoded_text[:i])
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.exp_rng)
            r = random_numbers[encoded_text[i]]
            v = log(1 / (1 - r))
            v = self.utils._value_transformation(v)
            highlight_values.append(v)

        # Decode each token id to its corresponding string token
        decoded_tokens = [self.config.generation_tokenizer.decode([token_id]) for token_id in encoded_text]

        return DataForVisualization(decoded_tokens, highlight_values)

