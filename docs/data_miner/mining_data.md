# 📦 data_miner – Procesador de vídeos LSM

Extrae señales visuales (keypoints de MediaPipe) y audio de vídeos de Lengua de Señas Mexicana (LSM) para alimentar el entrenador de modelos (`ai_trainer`).  
Puede trabajar con:

* URLs de YouTube (descarga automática)
* Archivos locales (`.mp4`, `.mov`, `.avi`, etc.)
* Segmentación fija (por defecto 2 s) o personalizada
* Transcripción opcional vía Whisper (si está instalado)

---

## 1. Instalación del ambiente

El módulo **debe ejecutarse en su propio entorno aislado** (`data_miner`).  
Usa el archivo `requirements/data_miner.txt` que ya fue exportado desde este venv.

```bash
# 1. Crear y activar entorno
python -m venv venv_dm
source venv_dm/bin/activate              # Linux / macOS
# o
venv_dm\Scripts\activate                 # Windows

# 2. Instalar dependencias
pip install -r requirements/data_miner.txt

# 3. Instalar groq mas la autentificacion con la api key para la utilizacion de whisper para el funcionamiento del modulo
https://console.groq.com/keys
crear un .env dentro del ambiente de "data_miner"
añadir en el ".env" la api key como "GROQ_API_KEY= XXXXx..."

