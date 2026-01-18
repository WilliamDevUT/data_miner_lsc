# config.md – Centro de control del proyecto

## 1. Propósito
Único punto de verdad para **ambientes** `data_miner` y `ai_trainer`.  
Cualquier script que importe `config.py` obtiene rutas, umbrales y utilidades sin hard-code.

## 2. Ubicación
repo/
├── config.py          ← este archivo
├── data_miner/
├── ai_trainer/
└── requirements/


## 3. Variables clave (solo lectura)

| Variable | Valor por defecto | Significado |
|----------|-------------------|-------------|
| `DATA_MINER_OUTPUT` | `./vid_data` | Videos procesados por `data_miner` |
| `TRAIN_DATA_DIR` | `./train_data` | Clips validados / desconocidos |
| `MODELS_DIR` | `./models` | Modelos `.keras` y `words.json` |
| `PREDICTION_THRESHOLD` | `0.7` | Confianza mínima para aceptar predicción |
| `MODEL_FRAMES` | `30` | Frames que espera el modelo |
| `SEGMENT_DURATION` | `2` | Segundos entre cortes en `data_miner` |

## 4. Funciones públicas (importables)

```python
from config import (
    ensure_directories,           # crea árbol de carpetas
    get_latest_video_dir,         # Path más reciente en vid_data
    get_model_paths,              # (model.keras, words.json) del último entrenamiento
    save_processing_state,        # json que comunica data_miner → predictor
    load_processing_state,        # lo lee el predictor
    validate_environment,         # chequea modelos y videos
) ```


## 5. Comunicación entre ambientes
data_miner finaliza con:
save_processing_state(video_dir, metadata)
predictor arranca con:
state = load_processing_state()   # obtiene rutas sin preguntar

## 6. Flujo típico
data_miner → vid_data/video_20260119_143022/...
save_processing_state() → last_processing_state.json
predictor --mode state → lee estado → predice
train.py → lee train_data/... → guarda modelo timestamped
Próxima predicción → get_model_paths() devuelve el último modelo

## 7. Extender sin romper
Para añadir un nuevo path o hiper-parámetro:
Añade la constante en config.py
Úsala en cualquier módulo que importe config
Commit → ambos ambientes lo ven instantáneamente
Nota: Nunca edites config.py dentro de un venv; siempre en la raíz del repo.
Con config.py el proyecto se comporta como un solo sistema a pesar de los dos ambientes.
