"""One-off data-prep script: turns finetune_data/claude_decisions_good.jsonl
(filtered by build_good_decisions.py to only good-outcome cycles) into a plain
{"system", "user", "assistant_tool_call"} JSONL, ready for a training script to
apply the base model's own chat template to (kept separate from that step so
the training venv never needs mt5_bot's own dependencies -- MetaTrader5,
anthropic, openai -- just torch/transformers/unsloth).

Uses the CURRENT SYSTEM_PROMPT/OPENAI_DECISIONS_TOOL (llm_strategy.py), not
whatever was live when each example was originally logged -- we want the
fine-tune to match today's prompt/schema, not an older version. Note: some
older examples predate fields added since (e.g. pivot/r1/s1) and simply won't
have those keys in their symbol_rows -- the prompt already treats a null/
absent pivot as "no extra structure context", so this is a minor imprecision,
not a correctness issue.

Not part of the live bot -- run manually:
`python -m mt5_bot.build_finetune_dataset` from the parent directory.
"""
import json
import os

from . import config, llm_strategy

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(_PROJECT_ROOT, "finetune_data", "claude_decisions_good.jsonl")
OUT_PATH = os.path.join(_PROJECT_ROOT, "finetune_data", "sft_dataset.jsonl")


def main():
    system_prompt = llm_strategy.SYSTEM_PROMPT.format(timeframe=config.TIMEFRAME)
    tool_schema = llm_strategy.OPENAI_DECISIONS_TOOL

    n = 0
    with open(SRC_PATH, encoding="utf-8") as src, open(OUT_PATH, "w", encoding="utf-8") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            user_content = (
                "ACCOUNT:\n" + json.dumps(ex["account_summary"]) +
                "\n\nSYMBOLS (only symbols with usable indicator data are included):\n" +
                json.dumps(ex["symbol_rows"])
            )
            if ex.get("headlines"):
                user_content += "\n\nNEWS (recent forex headlines, qualitative context only):\n" + json.dumps(ex["headlines"])

            record = {
                "system": system_prompt,
                "user": user_content,
                "tool_schema": tool_schema,
                "assistant_tool_call": {
                    "name": "trade_decisions",
                    "arguments": {"decisions": ex["decisions"]},
                },
            }
            out.write(json.dumps(record) + "\n")
            n += 1

    print(f"Wrote {n} training examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
