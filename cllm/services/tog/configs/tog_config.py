import os

import munch

from . import (
    resource_expert_prompts,
    solution_expert_prompts,
    task_solver_prompts,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

config = dict(
    memory=dict(max_tokens=-1),
    task_decomposer_cfg=dict(
        model=os.environ.get("TASK_DECOMPOSITION_CKPT", "OpenGVLab/cllm_td_opt")
    ),
    task_solver_config=dict(
        tog_cfg=dict(
            # strategy="greedy",
            # strategy="beam",
            strategy="adaptive",
            # strategy="exhaustive",
            tools=os.path.join(CURRENT_DIR, "tools.json"),
            prompts=task_solver_prompts.prompts,
        ),
        solution_expert_cfg=dict(
            prompts=solution_expert_prompts.prompts,
        ),
        resource_expert_cfg=dict(
            prompts=resource_expert_prompts.prompts,
        ),
    ),
)

config = munch.munchify(config)
