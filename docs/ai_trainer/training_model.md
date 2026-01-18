# train.md – Pipeline de entrenamiento LSM

## 1. Propósito
Convierte clips validados (`train_data/validadas/keypoints/&lt;palabra&gt;/*.json`) en un modelo `.keras` listo para predicción.

## 2. Instalación (entorno `ai_trainer`)
```bash
pip install -r requirements/ai_trainer.txt   # tensorflow, pandas, sklearn, etc.
```



## 3.probar

```python
training_model.py
```


Hace:
Descubre palabras → words.json
Convierte cada .json → .h5 (HDF5)
Entrena LSTM 128-256-128 → guarda modelo con timestamp

## 4. Salidas
```bash
ai_trainer/output/models/20260119_143022/
├── lstm_30f_20260119_143022.keras   # modelo
└── words.json                        # índices de clases
```

## 5. Opciones 

# Solo (re)entrenar (datos ya preparados)

```python
 training_model.py --no-prepare
```


# Más épocas / otro batch
```python
training_model.py --epochs 1000 --batch 16 --split 0.2
```


## 6. Datos esperados
```bash
train_data/
├── validadas/keypoints/hola/
│   ├── seña_1.json   # [[1662], [1662], ...]  (frames, keypoints)
│   └── seña_2.json
└── desconocidas/keypoints/adios/
    └── seña_3.json
```

## 7. arquitectura por defecto 

| Capa                      | Salida     |
| ------------------------- | ---------- |
| LSTM 128                  | (30, 1662) |
| LSTM 256                  | (30, 128)  |
| LSTM 128                  | (128,)     |
| Dense 128 → 64 → *clases* | softmax    |


