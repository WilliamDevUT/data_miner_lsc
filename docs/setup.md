
# 📦 Instalación completa – LSC-Trainer  
Guía paso a paso para Windows / macOS / Linux

---

## 0. Requisitos previos
- Python **3.10**  
- 4 GB de RAM mínimo (8 GB recomendado)  
- **ffmpeg** en el PATH (sin él no se extrae audio)  
- cuenta en [Groq](https://console.groq.com) → clave API para transcripción rápida

---

## 1. Clonar el repositorio
```bash
git clone https://github.com/tu_usuario/lsm-trainer.git
cd lsc-trainer
```
## 2. Instalar ffmpeg

| Sistema       | Comando                                                     |
| ------------- | ----------------------------------------------------------- |
| Windows       | `winget install ffmpeg` *(o descargar zip → añadir a PATH)* |
| macOS         | `brew install ffmpeg`                                       |
| Debian/Ubuntu | `sudo apt update && sudo apt install ffmpeg`                |

verificar:
```bash
ffmpeg -version
```
 
## 3. Crear los dos ambientes obligatorios
```bash
# Ambiente para extracción (data_miner)
python -m venv venv_dm

# Ambiente para entrenamiento (ai_trainer)
python -m venv venv_ai
```

## 4. Activar e instalar dependencias

### 1   data_miner
```bash
# Windows
venv_dm\Scripts\activate
# macOS/Linux
source venv_dm/bin/activate

pip install --upgrade pip
pip install -r requirements/data_miner.txt
```
### 2  ai_trainer
```bash
# Windows
venv_dm\Scripts\activate
# macOS/Linux
source venv_dm/bin/activate

pip install --upgrade pip
pip install -r requirements/data_miner.txt
```
## 5. Transcripción rápida con Groq
Regístrate en https://console.groq.com → API Keys
Crea .env en la raíz del repo:
`GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
Comprueba que groq_whisper.py lo detecta:
```bash
venv_dm\Scripts\activate
python -c "from groq_whisper import transcribe_audio_in_chunks; print('✓ Groq OK')"
```
## 6. Comprobar instalación
```bash
# data_miner
venv_dm\Scripts\activate
python data_miner/mining_data.py --help   # debe mostrar argumentos

# ai_trainer
venv_ai\Scripts\activate
python ai_trainer/train.py --help         # debe mostrar argumentos
   ```

Si los --help funcionan → ¡listo!

## 7. Prueba rápida (demo)

```bash

   # 1. Descarga un video corto de YouTube
venv_dm\Scripts\activate
python data_miner/mining_data.py --url "https://youtu.be/3rtF7_1xb9A" --duration 2

# 2. Entrena un mini-modelo (3-5 palabras de prueba)
venv_ai\Scripts\activate
python ai_trainer/train.py --epochs 10

# 3. Predice
python ai_trainer/predictor.py --mode latest
```
## 8. Solución de problemas comunes

| Error                    | Solución                                                                 |
| ------------------------ | ------------------------------------------------------------------------ |
| `ffmpeg not found`       | Añade la carpeta `bin` de ffmpeg al PATH y reinicia terminal             |
| `ImportError: cv2`       | `pip install opencv-python` dentro del ambiente activo                   |
| `Groq rate-limit`        | Añade `time.sleep(60)` en `groq_whisper.py` o usa `--skip-transcription` |
| `No module named config` | Ejecuta **desde la raíz del repo**, no desde sub-carpetas                |
| GPU no detectada         | TensorFlow CPU funciona igual; solo será más lento                       |

Lee el README principal para el flujo completo
Añade tus propios clips a train_data/validadas/keypoints/<palabra>/.
Re-entrena → mejora el modelo → valida → repite.
