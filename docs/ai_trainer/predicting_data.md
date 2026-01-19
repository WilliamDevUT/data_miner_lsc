# ai_trainer.md – Entrenamiento y predicción de señas

## 1. Propósito
Entrena modelos de clasificación de LSC (Lengua de Señas Colombiana) y predice palabras a partir de secuencias de keypoints generadas por `data_miner`.

## 2. Instalación (entorno `ai_trainer`)
```bash
python -m venv venv_ai
source venv_ai/bin/activate
pip install -r requirements/ai_trainer.txt   # tensorflow, keras, mediapipe, etc.
```
## 3. Pipeline rápido


### 1. Predecir último video procesado
```python predictor.py --mode latest```

### 2. Validar predicciones vs. Whisper y generar clips
```python predictor.py --mode validate --full-pipeline```


## 2. Salidas
```bash
ai_trainer/output/
├── models/
│   └── lstm_30f_2026-01-19_14-30.h5
├── predictions/
│   └── detecciones_20260119_143022.json
└── logs/
    └── training_20260119.log
```

## 3. Clases principales 

| Clase                        | Función                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| `SignPredictor`              | Carga modelo → predice sobre `sign.json` → devuelve lista de palabras con confianza   |
| `SignValidatorAndClipper`    | Compara predicciones vs. transcripción Whisper → corta clips validados y desconocidos |
| `VideoClipper`               | Corta clips con `moviepy` y numeración automática `seña_N.mp4`                        |
| `WhisperTranscriptionLoader` | Lee `audio.json` (words + timestamps) para validación                                 |


 
## 6. Integración con data_miner

Cuando config.py está presente:
Lee `MODEL_FRAMES`, `PREDICTION_THRESHOLD`, rutas de modelos.
Guarda `state.json` → predictor puede ejecutarse con `--mode state` sin indicar rutas.

## 7. validacion + clips
```bash
python predictor.py \
    --mode validate \
    --detections output/detecciones_xxx.json \
    --whisper vid_data/xxx/audio.json \
    --video vid_data/xxx/video.mp4

validated_videos/<palabra>/seña_1.mp4
validated_keypoints/<palabra>/seña_1.json
unknown_videos/...    
```
