# Copyright 2026 the verl recipe authors
"""
Validation reward for model **B** in ``MyTrainer._my_validate``: GSM8K (rule) + MBPP (execution).

Passed as ``compute_score`` to ``NaiveRewardManager`` (see ``main_ppo.py``).
"""

from __future__ import annotations

from recipe.my_project.mbpp_exec import compute_mbpp_score_dict

MBPP_DATA_SOURCE = "google-research-datasets/mbpp"


def val_b_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Dispatch by ``data_source``: MBPP uses subprocess execution of asserts; others defer to
    ``default_compute_score`` (e.g. ``openai/gsm8k``).

    ``default_compute_score`` is imported lazily so importing this module does not require Ray.
    """
    if data_source == MBPP_DATA_SOURCE:
        return compute_mbpp_score_dict(solution_str, ground_truth, extra_info)
    from verl.utils.reward_score import default_compute_score

    return default_compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        sandbox_fusion_url,
        concurrent_semaphore,
        memory_limit_mb,
    )
