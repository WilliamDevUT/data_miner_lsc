# LSC-Trainer  
**Reconocimiento de Lengua de Señas Colombiana con MediaPipe + TensorFlow**

---

## ¿Qué hace?
1. **Extrae** keypoints de video (MediaPipe)  
2. **Predice** palabras en tiempo real  
3. **Valida** las predicciones contra la transcripción (Whisper) y genera nuevos clips para re-entrenar
4. **Entrena** un modelo LSTM con los clips validados

## 📺 Videos de entrada esperados
Grabaciones de **noticieros o transmisiones en vivo** donde un **intérprete de LSM** traduzca la voz en tiempo real.  
El intérprete debe aparecer de medio cuerpo (de la cintura hacia arriba), de frente y sin cortes bruscos de cámara.
entre mejor sea la calidad del video mejor.

---

## Instalación rápida

```bash
# 1. Clona
git clone https://github.com/WilliamDevUT/data_miner_lsc.git
cd data_miner_lsc

## 2. Crea y activa los ambientes
python -m venv venv_dm      # data_miner
python -m venv venv_ai      # ai_trainer

# 3. Instala dependencias
venv_dm\Scripts\activate    # Linux/mac: source venv_dm/bin/activate
pip install -r requirements/data_miner.txt

venv_ai\Scripts\activate
pip install -r requirements/ai_trainer.txt
```
Mira setup.md para todos los pasos de instalación (ffmpeg, claves API, etc.).

(https://github.com/WilliamDevUT/data_miner_lsc/blob/main/docs/setup.md)



## Comandos tipicos 
| Paso | Comando                                                          | Descripción                                  |
| ---- | ---------------------------------------------------------------- | -------------------------------------------- |
| 1    | `python data_miner/mining_data.py --url YOU_TUBE_URL`            | Descarga → segmentos → keypoints             |
| 2    | `python ai_trainer/predictor.py --mode latest`                   | Predice último video                         |
| 3    | `python ai_trainer/predictor.py --mode validate --full-pipeline` | Corta clips buenos/malos y actualiza dataset |
| 4    | `python ai_trainer/training_model.py`                            | Entrena modelo (auto-guarda con fecha)       |

## Requisitos 
- Python 3.10
- ffmpeg (en PATH)
- Groq API key (opcional, para transcripción rápida)
