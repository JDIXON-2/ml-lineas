# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize

from lineas.evaluations import evaluate_0shot

SYSTEM_PROMPT = [
    "You are a chatbot that tells if a sentence is about fantasy.",
    "You are a chatbot that tells if a sentence is about football.",
]
ANSWERS = ["no", "yes"]
DATA = {
    "sentence": [
        "Hello, my name is John and I play a sport with a ball.",
        "Unicorns are horses with a horn and some magic.",
    ],
    "prompt": [
        "Can you introduce yourself?",
        "Dragons are like big lizards and breath fire.",
    ],
    "sentence2": [
        "Messi is the best player.",
        "I once saw a strange insect.",
    ],
}

SYSTEM_PROMPT_MC = [
    "Given the following 2 sentences, select the one that talks more about fantasy.",
    "Given the following 2 sentences, select the one that talks more about football.",
]
ANSWERS_MC = ["A", "B"]
DATA_MC = {
    "sentence": [
        "A: Hello, my name is John and I play a sport with a ball.",
        "A: Unicorns are horses with a horn and some magic.",
    ],
    "prompt": [
        "Can you introduce yourself?",
        "Dragons are like big lizards and breath fire.",
    ],
    "sentence2": [
        "B: Messi is the best player.",
        "B: I once saw a strange insect.",
    ],
}


# @pytest.mark.skip(reason="Uses Llama-3-8B-instruct, too large.")
@pytest.mark.parametrize(
    "system_prompt,answers,data,use_second_csv",
    [
        (SYSTEM_PROMPT, ANSWERS, DATA, False),
        (SYSTEM_PROMPT_MC, ANSWERS_MC, DATA_MC, True),
    ],
)
def test_0shot_e2e(system_prompt, answers, data, use_second_csv):
    # Assuming that the main function doesn't have any side effects and returns None when successful
    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        csv_file = Path(tempfolder) / "test.csv"
        df = pd.DataFrame(data=data)
        df.to_csv(csv_file)

        second_csv_argv = []
        if use_second_csv:
            csv_file2 = Path(tempfolder) / "test.csv"
            df.to_csv(csv_file2)
            second_csv_argv = [
                f"zero_shot.data_path2={csv_file2}",
                "zero_shot.col_sentence2=sentence2",
            ]

        with initialize(version_base=None, config_path="../lineas/configs"):
            # config is relative to a module

            cfg = compose(
                config_name="text_generation",
                overrides=[
                    "device=cpu",
                    "data_dir=tests/data",
                    "cache_dir=tests/data",
                    "wandb.mode=disabled",
                    f"zero_shot.system_prompt={system_prompt}",
                    f"zero_shot.system_answers={answers}",
                    "zero_shot.col_prompt=null",
                    "zero_shot.col_sentence1=sentence",
                    f"zero_shot.data_path={csv_file}",
                    f"zero_shot.results_dir={tempfolder}",
                    "zero_shot.model_path=Qwen/Qwen2.5-0.5B-Instruct",
                    *second_csv_argv,
                ],
            )

        evaluate_0shot.evaluate(cfg.zero_shot)
        df_out = pd.read_csv(Path(tempfolder) / "evaluate_0shot" / "0shot_eval.csv")

        assert len(df_out) == 2
        assert "q0_llm_answer" in df_out.columns
        assert "q1_llm_answer" in df_out.columns
        assert df_out["q0_llm_answer"].values[1] in ["yes", "A"]
