"""Prompt engineering module for document extraction.

Provides prompt templates, strategies, and an experiment runner for
evaluating different prompt approaches on Qwen2-VL.
"""

from src.prompts.templates import (
    FormExtractionTemplate,
    PromptTemplate,
    ReceiptExtractionTemplate,
    get_template,
    list_templates,
)
from src.prompts.strategies import (
    BaseStrategy,
    ChainOfThoughtSelfVerify,
    ChainOfThoughtStepByStep,
    FewShotStrategy2,
    FewShotStrategy5,
    ZeroShotBasic,
    ZeroShotDetailed,
    ZeroShotStructured,
    get_all_strategies,
    get_strategy,
    list_strategies,
)
from src.prompts.experiment_runner import PromptExperimentRunner

__all__ = [
    # Templates
    "PromptTemplate",
    "ReceiptExtractionTemplate",
    "FormExtractionTemplate",
    "get_template",
    "list_templates",
    # Strategies
    "BaseStrategy",
    "ZeroShotBasic",
    "ZeroShotDetailed",
    "ZeroShotStructured",
    "FewShotStrategy2",
    "FewShotStrategy5",
    "ChainOfThoughtStepByStep",
    "ChainOfThoughtSelfVerify",
    "get_strategy",
    "get_all_strategies",
    "list_strategies",
    # Runner
    "PromptExperimentRunner",
]
