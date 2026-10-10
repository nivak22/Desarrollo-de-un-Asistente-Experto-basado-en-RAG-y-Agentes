from __future__ import annotations

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

EVAL_DATA = [
    {
        "question": "¿Cuál es el voltaje nominal de la batería EvoBike Rayo Pro?",
        "ground_truth": "El voltaje nominal es de 72V según el manual de especificaciones.",
    },
    {
        "question": "¿Cómo se configura el BMS Daly para limitar la corriente?",
        "ground_truth": "Se debe acceder a la aplicación del BMS Daly, abrir parámetros de protección y ajustar la corriente máxima de descarga según el valor recomendado." ,
    },
    {
        "question": "¿Qué debe hacer si los frenos están muy largos?",
        "ground_truth": "Debe ajustar la tensión de la guaya del freno girando la tuerca de ajuste en sentido horario y verificar que la manigueta tenga 1 a 2 centímetros de recorrido libre.",
    },
    {
        "question": "¿Qué presión debe tener la llanta delantera y trasera?",
        "ground_truth": "La presión recomendada es 30 PSI en la llanta delantera y 32 PSI en la trasera.",
    },
    {
        "question": "¿Qué ocurre si la batería se descarga por debajo del 20%?",
        "ground_truth": "No se debe dejar descargar por debajo del 20% de su capacidad total porque se recomienda mantener un nivel de carga adecuado para prolongar la vida útil del sistema de almacenamiento.",
    },
    {
        "question": "¿Cuándo se recomienda guardar la batería por un periodo prolongado?",
        "ground_truth": "Si la motocicleta no se va a usar por más de un mes, se recomienda cargar la batería hasta el 80%, apagar el breaker principal y almacenarla en un lugar fresco y seco.",
    },
    {
        "question": "¿Qué debe revisar si la moto no acelera al girar el manillar?",
        "ground_truth": "Debe revisar el cableado principal, los sensores de corte y los sensores de freno para comprobar que no estén pellizcados, sueltos o pegados.",
    },
    {
        "question": "¿Puede cambiarse la batería de plomo-ácido por una de litio?",
        "ground_truth": "Sí, es posible cambiar la batería de plomo-ácido por una de litio, pero se debe desconectar el breaker, instalar la nueva batería con voltajes coincidentes y usar el cargador especial para litio.",
    },
    {
        "question": "¿Cuál es la velocidad máxima recomendada de la EvoBike Rayo Pro?",
        "ground_truth": "La velocidad máxima recomendada es de 45 km/h.",
    },
    {
        "question": "¿Qué debe hacer si la moto presenta vibraciones inusuales en la rueda trasera?",
        "ground_truth": "Podría indicar que los rodamientos necesitan lubricación o reemplazo, lo cual debe ser realizado por un centro de servicio autorizado.",
    },
    {
        "question": "¿Qué se debe revisar antes de cada recorrido?",
        "ground_truth": "Antes de cada recorrido se debe verificar el estado de la cadena, los frenos, los neumáticos y la carga del paquete de baterías.",
    },
    {
        "question": "¿Qué medida de seguridad se recomienda al almacenar la bicicleta?",
        "ground_truth": "Se recomienda mantener la batería con al menos el 50% de carga y desconectar el interruptor principal cuando se guarda la motocicleta.",
    },
    {
        "question": "¿Qué tan importante es la revisión periódica?",
        "ground_truth": "La revisión periódica y el mantenimiento preventivo son esenciales para prolongar la vida útil de la motocicleta y reducir fallas eléctricas.",
    },
    {
        "question": "¿Cuál es la capital de Francia?",
        "ground_truth": "La información no se encuentra en la base de conocimientos proporcionada.",
    },
    {
        "question": "¿Qué se debe hacer si el sistema activa protección por sobretemperatura?",
        "ground_truth": "Se debe permitir que el sistema se enfríe durante 10 a 15 minutos antes de continuar.",
    },
]


def run_evaluation():
    rag_chain = get_rag_chain()

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


if __name__ == "__main__":
    run_evaluation()
