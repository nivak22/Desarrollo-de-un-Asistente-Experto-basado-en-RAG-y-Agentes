# Asistente experto en manuales técnicos con RAG

## Descripción del proyecto
Este repositorio implementa un sistema de recuperación aumentada con generación (RAG) para responder preguntas sobre manuales técnicos de equipos y bicicletas eléctricas. La aplicación toma documentos del corpus, los divide en fragmentos, crea embeddings, almacena el índice vectorial y recupera los fragmentos más relevantes para responder con un modelo LLM.

La solución está desarrollada en Python con Streamlit, LangChain, ChromaDB y Google Gemini.

## Flujo completo del sistema RAG

### 1) Selección del corpus
Se define como corpus documental la carpeta `data/` del proyecto, donde se almacenan los manuales base en formatos PDF, DOCX, HTML, TXT y Markdown. La elección del corpus responde a la necesidad del asistente: contestar dudas sobre mantenimiento, especificaciones y procedimientos técnicos del equipo documentado.

La pertinencia del corpus se justifica porque:
- contiene la fuente autoritativa del dominio,
- está limitado al manual del producto y no a información externa no verificada,
- permite que la respuesta esté anclada a referencias reales y auditable.

### 2) Ingesta y extracción de texto
La ingesta extrae texto desde varios tipos de archivo:
- PDF con `PyPDFLoader`
- DOCX con `python-docx`
- HTML con `BeautifulSoup` / `HTMLParser`
- TXT y Markdown con `TextLoader`

La lógica se encuentra en `anteproyecto parte 2/src/build_index.py`.

### 3) Chunking
Se usa `RecursiveCharacterTextSplitter` con:
- `chunk_size = 1000`
- `chunk_overlap = 200`

Esto permite conservar contexto y evitar romper secciones importantes en fragmentos demasiado pequeños. El solapamiento preserva continuidad semántica en preguntas que requieren información secuencial o procedimental.

### 4) Vectorización
Se utiliza el modelo de embeddings `sentence-transformers/all-MiniLM-L6-v2` porque:
- ofrece buenas métricas de similitud y velocidad razonable,
- funciona bien para consultas de lenguaje natural,
- es adecuado para un corpus técnico de tamaño medio.

### 5) Base de datos vectorial
Se usa Chroma como almacén persistente local:
- se crea el índice en `chroma_db/`,
- se persisten los documentos fragmentados y sus metadatos,
- cada fragmento conserva información como `source`, `document_name`, `file_type`, `page` cuando está disponible.

Estos metadatos permiten citar la fuente que respalda cada respuesta.

### 6) Recuperación
La recuperación se realiza con `similarity_search(question, k=4)`. Se seleccionan los 4 fragmentos más similares a la pregunta y se pasan al LLM como contexto. Este valor ofrece un equilibrio entre relevancia y costo de contexto del modelo.

### 7) Generación
La generación se hace con `ChatGoogleGenerativeAI` usando el modelo configurable mediante la variable `GOOGLE_MODEL` (por defecto `gemini-3.8-flash`).

El sistema usa un prompt con estas reglas:
- responder solo con el contexto disponible,
- si no aparece la información, responder exactamente: `La información no se encuentra en la base de conocimientos proporcionada.`
- no inventar ni adivinar.

También se incorpora el historial de conversación en la llamada al modelo para admitir preguntas de continuidad y seguimiento.

## Evaluación con RAGAS
El proyecto incluye un script de evaluación en `anteproyecto parte 2/src/evaluate_rag.py`.

### Conjunto de evaluación
Se construyó un conjunto de 15 preguntas con su `ground_truth`, incluyendo casos sin respuesta en el corpus para medir control de alucinaciones.

### Métricas evaluadas
- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`

Los resultados se guardan en:
- `anteproyecto parte 2/resultados_ragas.csv`
- `anteproyecto parte 2/evaluacion_ragas.csv`

## Estructura del repositorio
```text
.
├── .gitignore
├── requirements.txt
├── README.md
├── anteproyecto parte 2/
│   ├── app.py
│   ├── .gitignore
│   ├── README.md
│   ├── data/
│   ├── chroma_db/        # generado al construir el índice
│   └── src/
│       ├── build_index.py
│       ├── evaluate_rag.py
│       ├── rag_core.py
│       └── __init__.py
└── documentos/
```

## Requisitos
- Python 3.13 o 3.12
- Clave API válida de Google Gemini
- Conexión a internet durante la primera ejecución para descargar dependencias y embeddings

## Instalación
Desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuración de entorno
Crear un archivo `.env` dentro de `anteproyecto parte 2/` con:

```dotenv
GOOGLE_API_KEY=TU_CLAVE_AQUI
GOOGLE_MODEL=gemini-3.8-flash
```

No subir la clave al repositorio ni a GitHub público.

## Construcción del índice
Desde la carpeta `anteproyecto parte 2`:

```powershell
python src/build_index.py
```

## Ejecución de la app
```powershell
cd "anteproyecto parte 2"
python -m streamlit run app.py
```

La app quedará disponible en `http://localhost:8501`.

## Despliegue en nube
Se puede desplegar en Streamlit Community Cloud o cualquier plataforma compatible con Python/Streamlit. El procedimiento recomendado es:

1. Subir este repositorio a GitHub.
2. Configurar las variables secretas en la plataforma de hosting (`GOOGLE_API_KEY`, `GOOGLE_MODEL`).
3. Seleccionar como archivo principal `anteproyecto parte 2/app.py`.
4. Hacer deploy de la app.

Importante: la clave de API no debe quedar embebida en el código ni en config pública.

## URL pública de la aplicación
No puedo publicar una URL real desde este entorno porque no tengo acceso a una cuenta de hosting ni a una plataforma de despliegue autenticada. La app está preparada para desplegarse en Streamlit Community Cloud o similar; la URL pública se generará una vez el repositorio esté conectado a la plataforma elegida.

## Nota sobre este entorno
Se corrigió la implementación para que el proyecto quede alineado con lo pedido por el profesor:
- ingreso de múltiples tipos de documentos,
- historial conversacional real,
- recuperación con metadatos de fuente,
- evaluación con RAGAS y dataset de 15 preguntas,
- documentación centralizada en este README.

## Siguientes mejoras recomendadas
- Añadir OCR para PDFs escaneados.
- Mejorar la selección del chunking según tipo de documento.
- Añadir versión con agentes y herramientas.
- Implementar trazabilidad más detallada del contexto recuperado.
- Medir latencia, costo de tokens y calidad por pregunta.
