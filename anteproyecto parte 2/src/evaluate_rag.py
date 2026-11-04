from __future__ import annotations

import csv
import sys
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_core import get_rag_chain

load_dotenv(PROJECT_ROOT / ".env")


def load_eval_data(path: Path | None = None) -> list[dict[str, str]]:
    dataset_path = path or PROJECT_ROOT / "eval_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el conjunto de evaluación: {dataset_path}")

    rows = []
    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({
                "question": row["question"].strip(),
                "ground_truth": row["ground_truth"].strip(),
            })
    return rows


EVAL_DATA = load_eval_data()


def _save_eval_csv(rows: list[dict], path: Path):
    fieldnames = ["question", "ground_truth", "answer", "context"]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation():
    rag_chain = get_rag_chain()

    rows = []
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("Generando respuestas del RAG para evaluación...")
    for item in EVAL_DATA:
        response = rag_chain.invoke({"input": item["question"], "chat_history": []})
        questions.append(item["question"])
        answers.append(response["answer"])
        contexts.append([doc.page_content for doc in response["context"]])
        ground_truths.append(item["ground_truth"])
        rows.append(
            {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": response["answer"],
                "context": "\n---\n".join(doc.page_content for doc in response["context"]),
            }
        )

    _save_eval_csv(rows, PROJECT_ROOT / "evaluacion_ragas.csv")

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    print("Ejecutando métricas Ragas...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    df = result.to_pandas()
    df.to_csv(PROJECT_ROOT / "resultados_ragas.csv", index=False)
    print("Evaluación completada. Resultados guardados en resultados_ragas.csv")
    print(result)

    return df


if __name__ == "__main__":
    run_evaluation()
