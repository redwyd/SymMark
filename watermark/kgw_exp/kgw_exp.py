import torch
import scipy
from math import log, sqrt
from functools import partial
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.compute_entropy import ComputeEntropy
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
from exceptions.exceptions import AlgorithmNameMismatchError
from visualize.data_for_visualization import DataForVisualization


class KgwExpConfig:
    """Config class for KgwExp algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the KgwExp configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/KgwExp.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'KgwExp':
            raise AlgorithmNameMismatchError('KgwExp', config_dict['algorithm_name'])

        # KGW
        self.gamma = config_dict['gamma']
        self.delta = config_dict['delta']
        self.kgw_hash_key = config_dict['kgw_hash_key']
        self.z_threshold = config_dict['z_threshold']
        self.kgw_prefix_length = config_dict['kgw_prefix_length']
        self.f_scheme = config_dict['f_scheme']
        self.window_scheme = config_dict['window_scheme']

        # EXP
        self.exp_prefix_length = config_dict['exp_prefix_length']
        self.exp_hash_key = config_dict['exp_hash_key']
        self.threshold = config_dict['threshold']
        self.sequence_length = config_dict['sequence_length']
        self.top_k = config_dict['top_k']

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


class KgwExpUtils:
    """Utility class for KgwExp algorithm, contains helper functions."""

    def __init__(self, config: KgwExpConfig, *args, **kwargs) -> None:
        """
            Initialize the KgwExp utility class.

            Parameters:
                config (KgwExpConfig): Configuration for the KgwExp algorithm.
        """
        self.config = config
        self.kgw_rng = torch.Generator(device=self.config.device)
        # self.kgw_rng.manual_seed(self.config.kgw_hash_key)
        self.kgw_rng.manual_seed(2971215073)
        self.prf = torch.randperm(self.config.vocab_size, device=self.config.device, generator=self.kgw_rng)
        self.f_scheme_map = {"time": self._f_time, "additive": self._f_additive, "skip": self._f_skip, "min": self._f_min}
        self.window_scheme_map = {"left": self._get_greenlist_ids_left, "self": self._get_greenlist_ids_self}

        self.exp_rng = torch.Generator()

    # KGW
    def _f(self, input_ids: torch.LongTensor) -> int:
        """Get the previous token."""
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))

    def _f_time(self, input_ids: torch.LongTensor):
        """Get the previous token time."""
        time_result = 1
        for i in range(0, self.config.kgw_prefix_length):
            time_result *= input_ids[-1 - i].item()
        return self.prf[time_result % self.config.vocab_size]

    def _f_additive(self, input_ids: torch.LongTensor):
        """Get the previous token additive."""
        # additive_result = 0
        # for i in range(0, self.config.kgw_prefix_length):
        #     additive_result += input_ids[-1 - i].item()
        # return self.prf[additive_result % self.config.vocab_size]
        return self.config.kgw_hash_key * input_ids[-1].sum().item()

    def _f_skip(self, input_ids: torch.LongTensor):
        """Get the previous token skip."""
        return self.prf[input_ids[- self.config.kgw_prefix_length].item()]

    def _f_min(self, input_ids: torch.LongTensor):
        """Get the previous token min."""
        return min(self.prf[input_ids[-1 - i].item()] for i in range(0, self.config.kgw_prefix_length))

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        return self.window_scheme_map[self.config.window_scheme](input_ids)

    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via leftHash scheme."""
        # self.kgw_rng.manual_seed((self.config.kgw_hash_key * self._f(input_ids)) % self.config.vocab_size)
        self.kgw_rng.manual_seed(self._f(input_ids) % (2 ** 64 - 1))
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.kgw_rng)
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids

    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via selfHash scheme."""
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        greenlist_ids = []
        f_x = self._f(input_ids)
        for k in range(0, self.config.vocab_size):
            h_k = f_x * int(self.prf[k])
            self.kgw_rng.manual_seed(h_k % self.config.vocab_size)
            vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.kgw_rng)
            temp_greenlist_ids = vocab_permutation[:greenlist_size]
            if k in temp_greenlist_ids:
                greenlist_ids.append(k)
        return greenlist_ids

    def _compute_z_score(self, observed_count: int, T: int) -> float:
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def calculate_entropy(self, model, tokenized_text: torch.Tensor):
        """Calculate entropy for each token in the tokenized_text."""
        with torch.no_grad():
            output = model(torch.unsqueeze(tokenized_text, 0)[:,-30:], return_dict=True)
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
            # token_entropy_list.insert(0, -10000.0)
            return token_entropy_list, semantic_entropy_list

    # Series
    def kgw_origin_score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        num_tokens_scored = len(input_ids) - self.config.kgw_prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.kgw_prefix_length} tokens required by the seeding scheme."
                )
            )

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.kgw_prefix_length)]

        for idx in range(self.config.kgw_prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)

        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags

    # Parallel
    def kgw_parallel_score_sequence(self, input_ids):
        """Score the input_ids based on 50/50."""
        num_tokens_scored = (len(input_ids) - self.config.kgw_prefix_length) / 2
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.kgw_prefix_length} tokens required by the seeding scheme."
                )
            )

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.kgw_prefix_length)]

        for idx in range(self.config.kgw_prefix_length, len(input_ids)):
            if idx % 2 == 0:
                curr_token = input_ids[idx]
                greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_flags.append(1)
                else:
                    green_token_flags.append(0)

        z_score_1 = self._compute_z_score(green_token_count, num_tokens_scored)

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.kgw_prefix_length)]

        for idx in range(self.config.kgw_prefix_length, len(input_ids)):
            if idx % 2 == 1:
                curr_token = input_ids[idx]
                greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_flags.append(1)
                else:
                    green_token_flags.append(0)

        z_score_2 = self._compute_z_score(green_token_count, num_tokens_scored)

        return z_score_1 if z_score_1 > z_score_2 else z_score_2

    # Hybrid
    def kgw_entropy_score_sequence(self, input_ids, entropy_list):
        """Score the input_ids based on the greenlist and entropy."""
        num_tokens_scored = (len(input_ids) - self.config.kgw_prefix_length -
                             len([e for e in entropy_list[self.config.kgw_prefix_length:] if
                                  e <= self.config.token_entropy_threshold]))
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                )
            )

        green_token_flags = [-1 for _ in range(self.config.kgw_prefix_length)]
        weights = [-1 for _ in range(self.config.kgw_prefix_length)]

        for idx in range(self.config.kgw_prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
            if entropy_list[idx] > self.config.token_entropy_threshold:
                weights.append(1)
            else:
                weights.append(0)

        # calculate number of green tokens where weight is 1
        green_token_count = sum(
            [1 for i in range(len(green_token_flags)) if green_token_flags[i] == 1 and weights[i] == 1])
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)

        return z_score, green_token_flags, weights

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
        num_scored = len(encoded_text) - self.config.exp_prefix_length
        total_score = 0

        for i in range(self.config.exp_prefix_length, len(encoded_text)):
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
        num_scored = (len(encoded_text) - self.config.exp_prefix_length) / 2
        total_score = 0

        for i in range(self.config.exp_prefix_length, len(encoded_text)):
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

        for i in range(self.config.exp_prefix_length, len(encoded_text)):
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
        num_tokens_scored = (len(input_ids) - self.config.exp_prefix_length -
                             len([e for e in entropy_list[self.config.exp_prefix_length:] if
                                  e >= self.config.semantic_entropy_threshold]))
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                )
            )

        green_token_flags = [-1 for _ in range(self.config.exp_prefix_length)]
        weights = [-1 for _ in range(self.config.exp_prefix_length)]

        total_score = 0

        for i in range(self.config.exp_prefix_length, len(input_ids)):
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



class KGWLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for KgwExp algorithm, process logits to add watermark."""

    def __init__(self, config: KgwExpConfig, utils: KgwExpUtils, *args, **kwargs) -> None:
        """
            Initialize the KGW logits processor.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
                utils (KGWUtils): Utility class for the KGW algorithm.
        """
        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(self, scores: torch.FloatTensor,
                             greenlist_token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor,
                               greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.kgw_prefix_length:
            return scores

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask,
                                             greenlist_bias=self.config.delta)
        return scores


class KgwExp(BaseWatermark):
    """Top-level class for the EXP algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = KgwExpConfig(algorithm_config, transformers_config)
        self.utils = KgwExpUtils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text using the EXP algorithm."""

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt",
                                                          add_special_tokens=True).to(self.config.device)

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
        # self.config.gen_kwargs['min_length'] = 1
        self.config.gen_kwargs['min_new_tokens'] = 1

        # Generate tokens
        for i in range(self.config.sequence_length):
            with torch.no_grad():
                if past:
                    output = self.config.generation_model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.config.generation_model(inputs)

            scores = output.logits[:, -1, :self.config.vocab_size]

            # 串联水印
            if args[0] == 'S':
                if inputs.shape[-1] >= self.config.kgw_prefix_length:

                    batched_greenlist_ids = [None for _ in range(inputs.shape[0])]

                    for b_idx in range(inputs.shape[0]):
                        greenlist_ids = self.utils.get_greenlist_ids(inputs[b_idx])
                        batched_greenlist_ids[b_idx] = greenlist_ids

                    green_tokens_mask = self.logits_processor._calc_greenlist_mask(
                        scores=scores,
                        greenlist_token_ids=batched_greenlist_ids
                    )

                    scores = self.logits_processor._bias_greenlist_logits(
                        scores=scores,
                        greenlist_mask=green_tokens_mask,
                        greenlist_bias=self.config.delta
                    )

                    # Get probabilities
                    probs = torch.nn.functional.softmax(scores, dim=-1).cpu()

                    # Generate r1, r2,..., rk
                    self.utils.seed_rng(inputs[0])
                    random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.exp_rng)

                    # Sample token to add watermark
                    token = self.utils.exp_sampling(probs, random_numbers).to(self.config.device)

            # 并联水印
            elif args[0] == 'P':
                if i % 2 == 0:
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

            # 混联水印
            elif args[0] == 'H':
                result = args[2].get_entropy(inputs[:, -200:])
                token_entropy, semantic_entropy = result['token_entropy'], result['semantic_entropy']
                # print(f"token_entropy:{token_entropy}, semantic_entropy:{semantic_entropy}")

                token_total_entropy += token_entropy
                semantic_total_entropy += semantic_entropy

                if token_entropy > self.config.token_entropy_threshold and semantic_entropy < self.config.semantic_entropy_threshold:
                    A += 1
                    if inputs.shape[-1] >= self.config.kgw_prefix_length:

                        batched_greenlist_ids = [None for _ in range(inputs.shape[0])]

                        for b_idx in range(inputs.shape[0]):
                            greenlist_ids = self.utils.get_greenlist_ids(inputs[b_idx])
                            batched_greenlist_ids[b_idx] = greenlist_ids

                        green_tokens_mask = self.logits_processor._calc_greenlist_mask(
                            scores=scores,
                            greenlist_token_ids=batched_greenlist_ids
                        )

                        scores = self.logits_processor._bias_greenlist_logits(
                            scores=scores,
                            greenlist_mask=green_tokens_mask,
                            greenlist_bias=self.config.delta
                        )

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
        # self.config.gen_kwargs['min_length'] = args[1] + 30
        self.config.gen_kwargs['min_new_tokens'] = args[1]

        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> dict:
        """Detect watermark in the text."""
        # Encode the text
        exp_encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        kgw_encoded_text = exp_encoded_text.to(self.config.device)

        if args[0] == 'S':
            # Compute z_score using a utility method
            z_score, _ = self.utils.kgw_origin_score_sequence(kgw_encoded_text)
            # Calculate the number of tokens to score, excluding the prefix
            p_value = self.utils.exp_origin_score_sequence(exp_encoded_text)

        elif args[0] == 'P':
            z_score = self.utils.kgw_parallel_score_sequence(kgw_encoded_text)
            p_value = self.utils.exp_parallel_score_sequence(exp_encoded_text)

            # z_score, _ = self.utils.kgw_origin_score_sequence(kgw_encoded_text)
            # p_value = self.utils.exp_origin_score_sequence(exp_encoded_text)

        else:
            # token_entropy_list, semantic_entropy_list = self.utils.calculate_entropy(model=self.config.generation_model, tokenized_text=kgw_encoded_text)
            # z_score, _, _ = self.utils.kgw_entropy_score_sequence(kgw_encoded_text, token_entropy_list)
            # p_value = self.utils.exp_entropy_score_sequence(exp_encoded_text, semantic_entropy_list)
            # print(f"Mean Token Entropy: {sum(token_entropy_list) / len(token_entropy_list)}, Mean Semantic Entropy: {sum(semantic_entropy_list) / len(semantic_entropy_list)}", end="")
            z_score, _ = self.utils.kgw_origin_score_sequence(kgw_encoded_text)
            p_value = self.utils.exp_origin_score_sequence(exp_encoded_text)

        # Determine if the z_score indicates a watermark
        is_kgw_watermarked = z_score > self.config.z_threshold

        # Determine if the computed score exceeds the threshold for watermarking
        is_exp_watermarked = p_value < self.config.threshold

        # kgw_p_value = 1 - scipy.stats.norm.cdf(z_score)
        # exp_p_value = p_value
        # is_watermarked = is_kgw_watermarked if kgw_p_value < exp_p_value else exp_p_value

        is_watermarked = bool(is_kgw_watermarked | is_exp_watermarked)

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": {'KGW': z_score, 'EXP': p_value}}
        else:
            return (is_watermarked, [z_score, p_value])

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
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.exp_rng)
            r = random_numbers[encoded_text[i]]
            v = log(1 / (1 - r))
            v = self.utils._value_transformation(v)
            highlight_values.append(v)

        # Decode each token id to its corresponding string token
        decoded_tokens = [self.config.generation_tokenizer.decode([token_id]) for token_id in encoded_text]

        return DataForVisualization(decoded_tokens, highlight_values)
