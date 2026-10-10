# Asistente Experto de Manuales Técnicos (RAG)

Profesor, en este proyecto desarrollé un asistente web que permite consultar manuales técnicos en lenguaje natural. Para responder, recupera fragmentos relevantes de los documentos cargados y los envía a Google Gemini como contexto. La aplicación también muestra las fuentes documentales utilizadas.

## Nota importante para ejecutar el proyecto

Por seguridad, no incluí mi clave de Google en los archivos del proyecto. Si ejecuta la aplicación localmente, deberá crear su propio `.env`; si la abre desde la URL desplegada, deberá ingresar su propia clave en el campo protegido de la página. Sin una clave válida para Gemini, no se podrán enviar preguntas al modelo.

## Estado y alcance

El proyecto implementa un flujo **RAG** (*Retrieval-Augmented Generation*):

1. Carga archivos PDF, TXT y Markdown desde `data/`.
2. Divide el contenido en fragmentos de 1000 caracteres, con 200 caracteres de solapamiento.
3. Genera embeddings con `sentence-transformers/all-MiniLM-L6-v2` y los almacena localmente con Chroma.
4. Recupera los 4 fragmentos más similares a cada pregunta.
5. Envía la pregunta y el contexto recuperado a Gemini, y presenta la respuesta y sus fuentes en Streamlit.

La versión que entrego implementa una cadena RAG sencilla; **no implementa agentes autónomos ni uso de herramientas**. Si la rúbrica requiere agentes, esta funcionalidad queda pendiente de desarrollo y no se presenta aquí como implementada.

## Requisitos

- Windows, macOS o Linux.
- Python 3.13 recomendado (el proyecto se verificó en Python 3.13.14).
- Su propia clave de API de Google válida para Gemini, ya sea en `.env` para uso local o en el campo protegido al usar la URL.
- Conexión a Internet durante la primera ejecución, para descargar dependencias y el modelo de embeddings, y para consultar Gemini.

## Estructura del proyecto

```text
.
├── app.py                 # Interfaz web de Streamlit
├── ../requirements.txt    # Dependencias de Python (raíz del repositorio)
├── data/                  # Manuales PDF, TXT o Markdown de entrada
├── chroma_db/             # Índice vectorial local generado (no se versiona)
└── src/
    ├── build_index.py     # Carga documentos y construye el índice
    ├── rag_core.py        # Recuperación de contexto y generación de respuestas
    └── evaluate_rag.py    # Evaluación experimental con RAGAS
```

## Instalación y configuración

Abra PowerShell en la carpeta del proyecto:

```powershell
cd "RUTA\AL\PROYECTO"
```

Recomiendo crear un entorno virtual e instalar allí las dependencias:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r ..\requirements.txt
```

Si ya tiene un entorno virtual activado, puede usar `python -m pip install -r ..\requirements.txt` en lugar de las dos últimas rutas.

### Configuración de su clave de Google

Para ejecutar el proyecto localmente, cree un archivo llamado `.env` en la carpeta del proyecto, junto a `app.py`, y añada **su propia clave**:

```dotenv
GOOGLE_API_KEY=SU_CLAVE_DE_GOOGLE
```

Sustituya `SU_CLAVE_DE_GOOGLE` por una clave válida, sin comillas. No incluya la clave en el repositorio ni la comparta en capturas o documentos. Yo no incluyo ninguna clave en esta entrega y el archivo `.env` está excluido del control de versiones mediante `.gitignore`.

## Añadir documentos y construir el índice

Coloque los manuales que desea consultar en `data/`. Se admiten `.pdf`, `.txt` y `.md`; los subdirectorios no se recorren.

Construya el índice desde PowerShell:

```powershell
.\.venv\Scripts\python.exe src\build_index.py
```

El primer uso puede tardar porque descarga el modelo de embeddings. El índice se guarda localmente en `chroma_db/`.

Si agrega, quita o reemplaza documentos después de crear el índice, reconstruya la base para que refleje el nuevo corpus. Para evitar conservar fragmentos antiguos, detenga Streamlit y elimine **solo la base generada** antes de reconstruir:

```powershell
Remove-Item -Recurse -Force .\chroma_db
.\.venv\Scripts\python.exe src\build_index.py
```

No elimine la carpeta `data/`: contiene los documentos originales.

## Ejecutar la aplicación

Desde la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abra en el navegador la dirección local que muestre Streamlit, normalmente `http://localhost:8501`. La primera carga puede tardar mientras se inicializan los embeddings. Para detener el servidor, vuelva a la terminal y pulse `Ctrl+C`.

Puede probar el asistente con preguntas sobre información que aparezca en los manuales. Si no se recupera contexto suficiente, está configurado para responder que la información no se encuentra en la base de conocimientos; esa respuesta no debe interpretarse como un diagnóstico técnico confirmado.

## Publicar una URL para el profesor

La aplicación se puede desplegar en Streamlit Community Cloud conectándola a un repositorio de GitHub. El profesor podrá abrir la URL y escribir su propia clave de Google Gemini en el campo protegido de la aplicación. La clave se usa para consultar Gemini durante la sesión; no se incluye en el código, GitHub ni la configuración del hosting. Al cerrar la sesión, deberá ingresarla de nuevo.

### Preparar el repositorio

Antes de desplegar:

1. Suba a GitHub el código de esta carpeta, incluidos `app.py`, `src/` y los documentos autorizados de `data/`, además de `requirements.txt` en la raíz del repositorio. Streamlit Cloud solo puede desplegar archivos que estén en el repositorio.
2. No suba `.env`, `.venv/` ni `chroma_db/`. Están excluidos por `.gitignore`; el índice se generará en el servidor a partir de los documentos incluidos en `data/`.
3. Compruebe que los manuales se pueden compartir en el repositorio y que no incluyen información confidencial.
4. Si no desea publicar el código o los manuales, no use un repositorio público: elija un repositorio privado y confirme que su cuenta de Streamlit Community Cloud puede desplegarlo.

### Crear el despliegue

1. Inicie sesión en [Streamlit Community Cloud](https://share.streamlit.io/) con GitHub y autorice el acceso al repositorio.
2. Cree una aplicación nueva y seleccione el repositorio, la rama que contiene el proyecto y el archivo principal `anteproyecto parte 2/app.py` (o `app.py` si publicó el contenido de esta carpeta en la raíz del repositorio).
3. Inicie el despliegue. La primera ejecución puede tardar porque el servidor instalará dependencias, descargará el modelo de embeddings y construirá el índice a partir de `data/`.
4. Cuando Streamlit muestre la dirección de la aplicación, ábrala y pruebe una pregunta cuya respuesta aparezca en los manuales.
5. Comparta esa dirección con el profesor. Para usar Gemini, deberá ingresar su propia clave en el campo **Clave de Google Gemini**. No es necesario añadir una clave en Streamlit Secrets para este modo de uso.

La URL solo funcionará mientras la aplicación esté desplegada y el repositorio tenga los archivos necesarios. En el plan gratuito, la aplicación puede suspenderse por inactividad y tardar en reactivarse cuando alguien vuelva a abrirla. Cualquier persona que pueda abrir la app podrá utilizarla con su propia clave.

## Evaluación experimental

Incluí `src/evaluate_rag.py` para generar respuestas a un conjunto de preguntas y calcular métricas RAGAS de fidelidad, relevancia de respuesta, precisión y recuperación del contexto. Para ejecutarlo:

```powershell
.\.venv\Scripts\python.exe src\evaluate_rag.py
```

El resultado se guarda como `resultados_ragas.csv` en la carpeta del proyecto. Antes de interpretar las métricas, recomiendo revisar las preguntas y respuestas esperadas de `EVAL_DATA` y confirmar que se correspondan con los manuales de `data/`. La evaluación llama al modelo y requiere que usted haya configurado su propia clave de Google y tenga conexión a Internet.

## Pruebas sugeridas para la entrega

Profesor, sugiero revisar el sistema con los siguientes casos:

1. Preguntas cuya respuesta aparezca explícitamente en los manuales.
2. Respuestas útiles y sus fuentes documentales y páginas recuperadas.
3. Preguntas que no estén cubiertas por los documentos, para comprobar que el asistente indique que no encuentra la información en lugar de inventarla.
4. Una tabla o registro de preguntas, respuestas observadas y fuentes consultadas como evidencia de la demostración.
5. Las métricas de evaluación, después de confirmar que el conjunto de referencia coincide con los manuales cargados.
6. Los requisitos de la rúbrica relacionados con agentes. La versión actual no incluye agentes autónomos ni uso de herramientas.

## Solución de problemas

- **`streamlit` no se reconoce:** ejecútelo como módulo: `.\.venv\Scripts\python.exe -m streamlit run app.py`.
- **No se puede consultar Gemini:** si ejecuta la versión local, revise que `.env` esté junto a `app.py` y use `GOOGLE_API_KEY`. Si usa la URL, ingrese una clave válida en el campo protegido.
- **No existe la base de conocimiento:** ejecute `src\build_index.py` antes de iniciar la aplicación.
- **La respuesta no encuentra información:** confirme que el manual esté en `data/` y reconstruya el índice después de agregarlo.
- **La respuesta muestra información inesperada:** revise los fragmentos desplegados en “Ver fuentes documentales” y verifique que la respuesta esté sustentada por esos fragmentos.
