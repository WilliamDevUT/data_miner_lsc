# constants.md – Diccionario de palabras (legacy)

## 1. Propósito
Solo mantiene el mapeo **ID → texto legible** para mostrar en pantalla o logs.  
*No* define rutas ni hiper-parámetros; eso ya lo hace `config.py`.

## 2. Uso real
Se importa en:
```python
from constants import words_text, word_ids
```
Ejemplo:

Python:
```python
palabra = word_ids[pred_idx]        # 'hola'
texto   = words_text['hola']        # 'HOLA'
```
## 3. ¿Se usa todavía?

Sí, pero solo como fallback visual.
El predictor prefiere words.json del modelo; si no existe, carga constants.words_text.

## 4. Por qué no se elimina
Scripts antiguos y notebooks lo referencian.
Permite ejecutar el predictor sin haber entrenado (modo demo).
No rompe nada y pesa 2 KB.

## 5. Regla de oro
Nunca edites rutas ni frames aquí; hazlo en config.py.
Si añades una nueva palabra, agrégala a words_text y asegúrate de que exista en train_data/ antes de entrenar.
Listo: convive con config sin conflictos.
