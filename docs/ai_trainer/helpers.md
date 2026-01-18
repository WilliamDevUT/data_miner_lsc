# helpers.md – Utilidades compartidas

## 1. Propósito
Funciones de *utility* usadas por `data_miner` y `ai_trainer` para:
- Detección MediaPipe
- Extracción / dibujo de keypoints
- Creación de carpetas
- Carga de secuencias para entrenamiento


## 2. Funciones principales 
| Nombre                               | Entrada                                     | Salida                           | Descripción                                        |
| ------------------------------------ | ------------------------------------------- | -------------------------------- | -------------------------------------------------  |
| `mediapipe_detection(image, model)`  | `cv2.Mat`, `mp.solutions.holistic.Holistic` | `NamedTuple`                     | Ejecuta el modelo y devuelve resultados            |
| `extract_keypoints(results)`         | resultados MediaPipe                        | `np.ndarray` 1-D (1-662 valores) | Concatena pose(33×4) + face(468×3) + hands(2×21×3) |
| `draw_keypoints(image, results)`     | `cv2.Mat`, resultados                       | None                             | Dibuja landmarks sobre la imagen                   |
| `create_folder(path)`                | `str` / `Path`                              | None                             | Crea árbol si no existe                            |
| `there_hand(results)`                | resultados MediaPipe                        | `bool`                           | `True` si hay alguna mano detectada                |
| `get_sequences_and_labels(words_id)` | lista de IDs                                | `sequences[], labels[]`          | Lee `.h5` de keypoints y devuelve listas para entrenar |



