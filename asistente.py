
import os
import json
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

# 1. Configuración inicial
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# 2. Carga y Fragmentación de Documentos (Chunking)
def cargar_documentos(ruta_carpeta):
    documentos = []
    if not os.path.exists(ruta_carpeta):
        os.makedirs(ruta_carpeta)
        return documentos
        
    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith(".txt"):
            with open(os.path.join(ruta_carpeta, archivo), 'r', encoding='utf-8') as f:
                contenido = f.read()
                # Separación básica para no perder contexto
                fragmentos = contenido.split('\n\n') 
                for frag in fragmentos:
                    if len(frag.strip()) > 30: 
                        documentos.append({"texto": frag.strip(), "fuente": archivo})
    return documentos

# 3. Indexado Local y Recuperación (RAG)
def recuperar_contexto(pregunta, documentos, top_k=3):
    if not documentos: return ""
    textos = [doc["texto"] for doc in documentos]
    textos.append(pregunta)
    
    # Vectorización y Similitud de Coseno local
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(textos)
    similitudes = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    indices_top = similitudes.argsort()[-top_k:][::-1]
    contexto_recuperado = ""
    for i in indices_top:
        if similitudes[i] > 0.05:
            doc = documentos[i]
            contexto_recuperado += f"[Fuente: {doc['fuente']}]\n{doc['texto']}\n\n"
    return contexto_recuperado

# 4. Prompting y Generación
def preguntar_asistente(pregunta, documentos):
    contexto = recuperar_contexto(pregunta, documentos)
    if not contexto:
        return json.dumps({"error": "La respuesta no está en el manual proporcionado."})
        
    SYSTEM_PROMPT = """Eres un Analista de Soporte Técnico experto en mopeds eléctricas EvoBike Rayo Pro. 
Tu única fuente de verdad son los fragmentos de manuales que se te entregan dentro de la etiqueta <contexto>.
REGLAS:
1. Responde SOLO usando la información del <contexto>. Si la respuesta no está allí, dilo explícitamente, no inventes información.
2. Sé claro, breve y técnico.
3. Siempre responde en el siguiente formato JSON, sin texto adicional:
{
  "respuesta": "explicación clara de la solución",
  "pasos": ["paso 1", "paso 2", "..."],
  "fuente": "nombre del archivo de donde sacaste la información",
  "confianza": "alta | media | baja"
}"""

    FEW_SHOT_EXAMPLES = """
<ejemplo>
Pregunta: "Los frenos están muy largos"
Respuesta:
{
  "respuesta": "Es necesario ajustar la tensión de la guaya del freno desde la tuerca de ajuste.",
  "pasos": ["Ubicar la tuerca de ajuste en el tambor de freno", "Girar la tuerca en sentido horario para tensar"],
  "fuente": "manual_mantenimiento.txt",
  "confianza": "alta"
}
</ejemplo>
"""
    prompt_final = f"{SYSTEM_PROMPT}\n{FEW_SHOT_EXAMPLES}\n<contexto>\n{contexto}\n</contexto>\n<pregunta_usuario>\n{pregunta}\n</pregunta_usuario>"
    
    try:
        # Aquí actualizamos al modelo gemini-3.6-flash que te pidió la consola
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt_final,
            config=types.GenerateContentConfig(temperature=0.1)
        )
        return response.text.replace('```json', '').replace('```', '').strip()
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    print("Cargando documentos...")
    docs = cargar_documentos("documentos")
    print(f"Total de fragmentos: {len(docs)}")
    print("\n--- Asistente de Soporte Técnico EvoBike ---")
    
    while True:
        preg = input("\nEscribe tu pregunta (o 'salir'): ")
        if preg.lower() == 'salir':
            break
        print(preguntar_asistente(preg, docs))