# groq_whisper.md – Transcripción con word-level timestamps vía Groq

## 1. Propósito
Sustituye Whisper local por la API de Groq (`whisper-large-v3`) obteniendo **timestamps por palabra** sin GPU.

## 2. Requisitos previos
- Entorno `data_miner` ya creado e instalado (ver `requirements/data_miner.txt`).  
- Variable `GROQ_API_KEY` en `.env` raíz:
- https://console.groq.com/keys
-Asegúrate de tener ffmpeg en el PATH (usa el mismo que moviepy).

## 4. Salidas

Dentro de final_output_path siempre se generan

| Archivo      | Contenido                                  |
| ------------ | ------------------------------------------ |
| `audio.txt`  | Transcripción plana                        |
| `audio.json` | Dict completo: `text`, `segments`, `words` |


## 4. Funciones principales

| Función                     | Propósito                                             |
| --------------------------- | ----------------------------------------------------- |
| `preprocess_audio()`        | Convierte a 16 kHz mono FLAC (subida 30 % más rápida) |
| `transcribe_single_chunk()` | Llama a Groq con reintentos por rate-limit            |
| `merge_transcripts()`       | Alinea y fusiona solapamentos (LCS)                   |
| `save_results()`            | Guarda `.txt` y `.json` con nombre fijo `audio.*`     |


## 5. Posibles problemas 

| Error                       | Solución                                   |
| --------------------------- | ------------------------------------------ |
| `GROQ_API_KEY not set`      | Añade la variable al `.env` o shell        |
| `FFmpeg conversion failed`  | Instala `ffmpeg` y reinicia terminal       |
| `RateLimitError` continuo   | Aumenta `time.sleep(60)` o contacta a Groq |
| Archivos `.flac` temporales | Se borran automáticamente en `finally:`    |

## 6. Integración con data_miner

El módulo mining_data.py detecta automáticamente si groq_whisper está disponible
Si está presente, la transcripción se ejecuta dentro del pipeline sin 
configuración extra.


try:
    from groq_whisper import transcribe_audio_in_chunks
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
----------------------------------------------------------------------



