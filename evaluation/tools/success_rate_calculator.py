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

# ==========================================================
# success_rate_calculator.py
# Description: Calculate success rate of watermark detection
# ==========================================================

import scipy
import numpy as np

from typing import List, Dict, Union
from exceptions.exceptions import TypeMismatchException, ConfigurationError
from sklearn.metrics import roc_auc_score, roc_curve

class DetectionResult:
    """Detection result."""

    def __init__(self, gold_label: bool, detect_result: Union[bool, float]) -> None:
        """
            Initialize the detection result.

            Parameters:
                gold_label (bool): The expected watermark presence.
                detect_result (Union[bool, float]): The detection result.
        """
        self.gold_label = gold_label
        self.detect_result = detect_result


class BaseSuccessRateCalculator:
    """Base class for success rate calculator."""

    def __init__(self, labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']) -> None:
        """
            Initialize the success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
        """
        self.labels = labels

    def _check_instance(self, data: List[Union[bool, float]], expected_type: type):
        """Check if the data is an instance of the expected type."""
        for d in data:
            if not isinstance(d, expected_type):
                raise TypeMismatchException(expected_type, type(d))

    def _filter_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Filter metrics based on the provided labels."""
        return {label: metrics[label] for label in self.labels if label in metrics}

    def calculate(self, watermarked_result: List[Union[bool, float, list, dict]],
                  non_watermarked_result: List[Union[bool, float, list, dict]]) -> Dict[str, float]:
        """Calculate success rates based on provided results."""
        pass


class FundamentalSuccessRateCalculator(BaseSuccessRateCalculator):
    """
        Calculator for fundamental success rates of watermark detection.

        This class specifically handles the calculation of success rates for scenarios involving
        watermark detection after fixed thresholding. It provides metrics based on comparisons
        between expected watermarked results and actual detection outputs.

        Use this class when you need to evaluate the effectiveness of watermark detection algorithms
        under fixed thresholding conditions.
    """

    def __init__(self, labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']) -> None:
        """
            Initialize the fundamental success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
        """
        super().__init__(labels)

    @staticmethod
    def compute_metrics(inputs: List[DetectionResult]) -> Dict[str, float]:
        """Compute metrics based on the provided inputs."""
        TP = sum(1 for d in inputs if d.detect_result and d.gold_label)
        TN = sum(1 for d in inputs if not d.detect_result and not d.gold_label)
        FP = sum(1 for d in inputs if d.detect_result and not d.gold_label)
        FN = sum(1 for d in inputs if not d.detect_result and d.gold_label)

        TPR = TP / (TP + FN) if TP + FN else 0.0
        FPR = FP / (FP + TN) if FP + TN else 0.0
        TNR = TN / (TN + FP) if TN + FP else 0.0
        FNR = FN / (FN + TP) if FN + TP else 0.0
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * (P * R) / (P + R) if P + R else 0.0
        ACC = (TP + TN) / (len(inputs)) if inputs else 0.0

        return {
            'TPR': TPR, 'TNR': TNR, 'FPR': FPR, 'FNR': FNR,
            'P': P, 'R': R, 'F1': F1, 'ACC': ACC
        }

    def calculate(self, watermarked_result: List[bool], non_watermarked_result: List[bool]) -> Dict[str, float]:
        """calculate success rates of watermark detection based on provided results."""
        self._check_instance(watermarked_result, bool)
        self._check_instance(non_watermarked_result, bool)

        inputs = [DetectionResult(True, x) for x in watermarked_result] + [DetectionResult(False, x) for x in non_watermarked_result]
        metrics = self.compute_metrics(inputs)
        return self._filter_metrics(metrics)


class DynamicThresholdSuccessRateCalculator(BaseSuccessRateCalculator):
    """
        Calculator for success rates of watermark detection with dynamic thresholding.

        This class calculates success rates for watermark detection scenarios where the detection
        thresholds can dynamically change based on varying conditions. It supports evaluating the
        effectiveness of watermark detection algorithms that adapt to different signal or noise conditions.

        Use this class to evaluate detection systems where the threshold for detecting a watermark
        is not fixed and can vary.
    """

    def __init__(self,
                 labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'],
                 rule='best',
                 target_fpr=None,
                 reverse=False) -> None:
        """
            Initialize the dynamic threshold success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
                rule (str): The rule for determining the threshold. Choose from 'best' or 'target_fpr'.
                target_fpr (float): The target false positive rate to achieve.
                reverse (bool): Whether to reverse the sorting order of the detection results.
                                True: higher values are considered positive.
                                False: lower values are considered positive.
        """
        super().__init__(labels)
        self.rule = rule
        self.target_fpr = target_fpr
        self.reverse = reverse

        # Validate rule configuration
        if self.rule not in ['best', 'target_fpr']:
            raise ConfigurationError(f"Invalid rule specified: {self.rule}. Choose from 'best' or 'target_fpr'.")

        # Validate target_fpr configuration based on the rule
        if self.rule == 'target_fpr':
            if self.target_fpr is None:
                raise ConfigurationError("target_fpr must be set when rule is 'target_fpr'.")
            if not isinstance(self.target_fpr, (float, int)) or not (0 <= self.target_fpr <= 1):
                raise ConfigurationError("target_fpr must be a float or int within the range [0, 1].")

    def _find_best_threshold(self, inputs: List[DetectionResult]) -> float:
        """Find the best threshold that maximizes F1."""
        best_threshold = 0
        best_metrics = None
        for i in range(len(inputs) - 1):
            threshold = (inputs[i].detect_result + inputs[i + 1].detect_result) / 2
            metrics = self._compute_metrics(inputs, threshold)
            if best_metrics is None or metrics['F1'] > best_metrics['F1']:
                best_threshold = threshold
                best_metrics = metrics
        return best_threshold

    def _find_threshold_by_fpr(self, inputs: List[DetectionResult]) -> float:
        """Find the threshold that achieves the target FPR."""
        threshold = 0
        for i in range(len(inputs) - 1):
            threshold = (inputs[i].detect_result + inputs[i + 1].detect_result) / 2
            metrics = self._compute_metrics(inputs, threshold)
            if metrics['FPR'] <= self.target_fpr:
                break
        return threshold

    def _find_threshold(self, inputs: List[DetectionResult]) -> float:
        """Find the threshold based on the specified rule."""
        sorted_inputs = sorted(inputs, key=lambda x: x.detect_result, reverse=self.reverse)

        # If the rule is to find the best threshold by maximizing accuracy
        if self.rule == 'best':
            return self._find_best_threshold(sorted_inputs)
        else:
            # If the rule is to find the threshold that achieves the target FPR
            return self._find_threshold_by_fpr(sorted_inputs)

    def _compute_fpr_tpr_auroc_v1(self, inputs: List[DetectionResult]) -> [List, List, float]:
        # Calculate the AUROC
        labels = np.array([])
        scores = np.array([])
        if not self.reverse:
            for x in inputs:
                labels = np.append(labels, x.gold_label)
                scores = np.append(scores, x.detect_result)
        else:
            for x in inputs:
                labels = np.append(labels, 1 - x.gold_label)
                scores = np.append(scores, x.detect_result)
        scores[np.isnan(scores)] = 0
        auroc = roc_auc_score(labels, scores)
        fprs, tprs, thresholds = roc_curve(labels, scores)
        return fprs, tprs, float(auroc), thresholds

    def merge_roc(self, fpr1, tpr1, fpr2, tpr2):
        # 合并两组数据，并按FPR排序
        combined = sorted(list(zip(fpr1, tpr1)) + list(zip(fpr2, tpr2)))

        # 初始化合并后的FPR和TPR
        final_fpr = [combined[0][0]]
        final_tpr = [combined[0][1]]

        # 保证TPR递增
        for i in range(1, len(combined)):
            fpr, tpr = combined[i]

            # 确保每个FPR值对应的TPR值递增
            if tpr >= final_tpr[-1]:
                final_fpr.append(fpr)
                final_tpr.append(tpr)
            else:
                final_fpr.append(fpr)
                final_tpr.append(final_tpr[-1])  # 保持TPR递增，设置为上一个点的TPR

        # 计算AUROC
        auroc = np.trapz(final_tpr, final_fpr)  # 使用梯形规则计算面积

        # 确保首尾为[0, 0]和[1, 1]
        if final_fpr[0] != 0:
            final_fpr.insert(0, 0)
            final_tpr.insert(0, 0)
        if final_fpr[-1] != 1:
            final_fpr.append(1)
            final_tpr.append(1)

        return final_fpr, final_tpr, auroc

    def _compute_metrics(self, inputs: List[DetectionResult], threshold: float) -> Dict[str, float]:
        """Compute metrics based on the provided inputs and threshold."""
        if not self.reverse:
            TP = sum(1 for x in inputs if x.detect_result >= threshold and x.gold_label)
            FP = sum(1 for x in inputs if x.detect_result >= threshold and not x.gold_label)
            TN = sum(1 for x in inputs if x.detect_result < threshold and not x.gold_label)
            FN = sum(1 for x in inputs if x.detect_result < threshold and x.gold_label)
        else:
            TP = sum(1 for x in inputs if x.detect_result <= threshold and x.gold_label)
            FP = sum(1 for x in inputs if x.detect_result <= threshold and not x.gold_label)
            TN = sum(1 for x in inputs if x.detect_result > threshold and not x.gold_label)
            FN = sum(1 for x in inputs if x.detect_result > threshold and x.gold_label)

        FPRs, TPRs, AUROC, _ = self._compute_fpr_tpr_auroc_v1(inputs)

        metrics = {
            'TPR': TP / (TP + FN) if TP + FN else 0,
            'FPR': FP / (FP + TN) if FP + TN else 0,
            'TNR': TN / (TN + FP) if TN + FP else 0,
            'FNR': FN / (FN + TP) if FN + TP else 0,
            'P': TP / (TP + FP) if TP + FP else 0,
            'R': TP / (TP + FN) if TP + FN else 0,
            'F1': 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0,
            'ACC': (TP + TN) / (len(inputs)) if inputs else 0,
            'AUROC': AUROC,
            'FPRs': FPRs.tolist(),
            'TPRs': TPRs.tolist(),
        }
        return metrics

    def calculate(self, watermarked_result: List[float], non_watermarked_result: List[float], watermark_type='I') -> Dict[str, float]:
        """Calculate success rates based on provided results."""
        if watermark_type == 'I':
            self._check_instance(watermarked_result + non_watermarked_result, float)

            inputs = [DetectionResult(True, x) for x in watermarked_result] + [DetectionResult(False, x) for x in non_watermarked_result]
            threshold = self._find_threshold(inputs)
            metrics = self._compute_metrics(inputs, threshold)
        else:
            self._check_instance(watermarked_result + non_watermarked_result, dict)
            a = list(watermarked_result[0].keys())[0]
            b = list(watermarked_result[0].keys())[1]

            a_inputs = ([DetectionResult(True, x[a]) for x in watermarked_result] +
                        [DetectionResult(False, x[a]) for x in non_watermarked_result])
            b_inputs = ([DetectionResult(True, x[b]) for x in watermarked_result] +
                        [DetectionResult(False, x[b]) for x in non_watermarked_result])

            flags = [0 if 1 - scipy.stats.norm.cdf(x[a]) < x[b] else 1 for x in watermarked_result] + [0 if 1 - scipy.stats.norm.cdf(x[a]) < x[b] else 1 for x in non_watermarked_result]

            # kgw_p_value = 1 - scipy.stats.norm.cdf(z_score)
            # exp_p_value = p_value
            # is_watermarked = is_kgw_watermarked if kgw_p_value < exp_p_value else exp_p_value

            if self.rule == 'best':
                self.reverse = True if 'EXP' in a or 'Gumbel' in a else False
                a_threshold = self._find_threshold(a_inputs)
                print(f"\033[34ma_threshold: {a_threshold}\033[0m")
                a_threshold = 4 if a_threshold < 4 else a_threshold
                a_metrics = self._filter_metrics(self._compute_metrics(a_inputs, a_threshold))
                a_result = [True if x.detect_result >= a_threshold else False for x in a_inputs]

                self.reverse = True if 'EXP' in b or 'Gumbel' in b else False
                b_threshold = self._find_threshold(b_inputs)
                print(f"\033[34mb_threshold: {b_threshold}\033[0m")
                b_threshold = 1e-4 if b_threshold > 1e-4 else b_threshold
                b_metrics = self._filter_metrics(self._compute_metrics(b_inputs, b_threshold))
                b_result = [True if x.detect_result <= b_threshold else False for x in b_inputs]

                watermark_combine_result = [x | y for x, y in zip(a_result, b_result)][:len(watermarked_result)]
                unwatermarked_combine_result = [x | y for x, y in zip(a_result, b_result)][len(watermarked_result):]

                # watermark_combine_result = [x & y for x, y in zip(a_result, b_result)][:len(watermarked_result)]
                # unwatermarked_combine_result = [x & y for x, y in zip(a_result, b_result)][len(watermarked_result):]

                # watermark_combine_result = [y if z else x for x, y, z in zip(a_result, b_result, flags)][:len(watermarked_result)]
                # unwatermarked_combine_result = [y if z else x for x, y, z in zip(a_result, b_result, flags)][len(watermarked_result):]

                inputs = ([DetectionResult(True, x) for x in watermark_combine_result] +
                          [DetectionResult(False, x) for x in unwatermarked_combine_result])

                print(f"\033[38m{a}: {a_metrics}\033[0m")
                print(f"\033[38m{b}: {b_metrics}\033[0m")

                metrics = FundamentalSuccessRateCalculator.compute_metrics(inputs)

                if a_metrics['AUROC'] > b_metrics['AUROC']:
                    # metrics['FPRs'], metrics['TPRs'], metrics['AUROC'] = a_metrics['FPRs'], a_metrics['TPRs'], a_metrics['AUROC']
                    metrics['AUROC'] = a_metrics['AUROC']
                else:
                    # metrics['FPRs'], metrics['TPRs'], metrics['AUROC'] = b_metrics['FPRs'], b_metrics['TPRs'], b_metrics['AUROC']
                    metrics['AUROC'] = b_metrics['AUROC']

                # metrics['FPRs'], metrics['TPRs'], metrics['AUROC'] = self.merge_roc(a_metrics['FPRs'], a_metrics['TPRs'], b_metrics['FPRs'], b_metrics['TPRs'])

        return self._filter_metrics(metrics)
