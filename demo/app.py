"""Minimal interactive demo (plan item 10).

Replays 13 hand-picked question/answer pairs from the committed n=250
error-analysis run (results/error_analysis_250/) through a small Gradio UI.
No MongoDB Atlas, Voyage, Cohere, or Claude API calls are made - this is a
static replay of already-computed, already-audited pipeline output, not a
live pipeline run. See demo/data.py for exactly which questions were picked
and why, and README.md, "Minimal demo", for the scope this does and doesn't
cover.

Run (from the repo root, matching the pipeline's own `python -m` convention):
    python -m demo.app
Requires: pip install -r requirements-demo.txt (gradio; the rest of the
pipeline's own dependencies are not needed to run this demo).
"""

from __future__ import annotations

import gradio as gr

from demo.data import load_demo_sample

SAMPLE = load_demo_sample()
CHOICES = [f"{row['question_id']}  ({row['source_dataset']})" for row in SAMPLE]
BY_CHOICE = dict(zip(CHOICES, SAMPLE))


def render(choice: str) -> tuple[str, str, str, str, str, str]:
    row = BY_CHOICE[choice]
    verdict = "✅ correct" if row["judge_correct"] else "❌ incorrect"
    det = "match" if row["deterministic_match"] else "no match"
    agreement = (
        "agree"
        if row["judge_correct"] == row["deterministic_match"]
        else "DISAGREE — see docs/tehnicheskoe_zadanie.md, section 14"
    )
    return (
        row["source_dataset"],
        row["question"],
        row["gold_answer"],
        row["answer_text"],
        f"{row['judge_verdict']} ({verdict})",
        f"is_close_v2: {det}  |  judge vs. deterministic: {agreement}",
    )


with gr.Blocks(title="Financial RAG Pipeline — minimal demo") as demo:
    gr.Markdown(
        "# Financial RAG Pipeline — minimal demo\n\n"
        "Replays 13 hand-picked questions from the committed n=250 run "
        "(`results/error_analysis_250/`) — real questions, real gold answers, "
        "real model output, real judge verdicts. **No live API calls**: this "
        "is a static replay for browsing the pipeline's actual output, not a "
        "live retrieval/generation demo. 6 are judge-correct baseline "
        "examples across the three source datasets; 7 are documented failure "
        "or judge/deterministic-disagreement cases from "
        "`docs/tehnicheskoe_zadanie.md`, section 14. Full context: README.md, "
        "\"Minimal demo\"."
    )
    picker = gr.Dropdown(choices=CHOICES, value=CHOICES[0], label="Question")
    with gr.Row():
        source_out = gr.Textbox(label="source_dataset")
        judge_out = gr.Textbox(label="Judge verdict")
    question_out = gr.Textbox(label="Question", lines=2)
    with gr.Row():
        gold_out = gr.Textbox(label="Gold answer")
        answer_out = gr.Textbox(label="Model answer")
    agreement_out = gr.Textbox(label="Deterministic check")

    picker.change(
        render,
        inputs=picker,
        outputs=[source_out, question_out, gold_out, answer_out, judge_out, agreement_out],
    )
    demo.load(
        render,
        inputs=picker,
        outputs=[source_out, question_out, gold_out, answer_out, judge_out, agreement_out],
    )


if __name__ == "__main__":
    demo.launch()
