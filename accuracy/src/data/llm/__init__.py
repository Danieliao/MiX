from .hf import *

DATA_STAGE_MAP = {
    "wikitext": WikiText,
    "c4": C4,
    "piqa": PiQA,
    "hellaswag": HellaSwag,
    "boolq": BoolQ,
    "openbookqa": OpenBookQA,
    "winogrande": WinoGrande,
    "ARC-Challenge": ARCc,
    "ARC-Easy": ARCe,
    "gsm8k": GSM8K,
    "mmlu": MMLU,
    "math": MATH500Step
}