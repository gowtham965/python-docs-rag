import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from pyrag.eval.retrieval_metrics import hit_rate_at_k, mean_reciprocal_rank
from pyrag.eval.judge import judge_answer


@dataclass
class EvalRow:
    question: str
    hit_at_5: bool
    mrr: float
    faithfulness: int
    relevance: int


def load_questions(path: str) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_eval(
    questions_path: str,
    retriever,
    pipeline,
    judge_llm_client,
    report_out_path: str,
) -> List[EvalRow]:
    questions = load_questions(questions_path)
    rows: List[EvalRow] = []

    for item in questions:
        if not item["is_in_scope"]:
            continue

        retrieval = retriever.retrieve(item["question"])
        source_files = [rc.chunk.source_file for rc in retrieval.chunks]

        hit = hit_rate_at_k(source_files, item["expected_source_file"], k=5)
        mrr = mean_reciprocal_rank(source_files, item["expected_source_file"])

        result = pipeline.answer(item["question"])
        score = judge_answer(item["question"], result.answer, result.sources, judge_llm_client)

        rows.append(
            EvalRow(
                question=item["question"],
                hit_at_5=hit,
                mrr=mrr,
                faithfulness=score.faithfulness,
                relevance=score.relevance,
            )
        )

    Path(report_out_path).write_text(
        json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
    )
    return rows
