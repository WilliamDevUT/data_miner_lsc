# LSM-Trainer  
**Reconocimiento de Lengua de Señas Colombiana con MediaPipe + TensorFlow**

---

## ¿Qué hace?
1. **Extrae** keypoints de video (MediaPipe)  
2. **Entrena** un modelo LSTM con los clips validados  
3. **Predice** palabras en tiempo real  
4. **Valida** las predicciones contra la transcripción (Whisper) y genera nuevos clips para re-entrenar

---

## Instalación rápida

```bash
# 1. Clona
git clone https://github.com/tu_usuario/lsm-trainer.git
cd lsm-trainer

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

<!-- docs/setup.md -->
<meta http-equiv="refresh" content="0"; url="https://github.com/WilliamDevUT/data_miner_lsc/blob/main/docs/setup.md"/>

<!-- docs/setup.md -->
<meta http-equiv="refresh" content="0; url="/>
<button onclick="location.href='https://github.com/WilliamDevUT/data_miner_lsc/blob/main/docs/setup.md">Llévame a otro lado</button>
<a>sasas</a>
[![Setup](https://img.shields.io/badge/📘%20Setup-Click%20here-blue)](https://github.com/WilliamDevUT/data_miner_lsc/blob/main/docs/setup.md)
[![Setup](https://raw.githubusercontent.com/github/explore/main/topics/markdown/markdown.png)](https://github.com/WilliamDevUT/data_miner_lsc/blob/main/docs/setup.md)



## Comandos tipicos 
| Paso | Comando                                                          | Descripción                                  |
| ---- | ---------------------------------------------------------------- | -------------------------------------------- |
| 1    | `python data_miner/mining_data.py --url YOU_TUBE_URL`            | Descarga → segmentos → keypoints             |
| 2    | `python ai_trainer/training_model.py`                            | Entrena modelo (auto-guarda con fecha)       |
| 3    | `python ai_trainer/predictor.py --mode latest`                   | Predice último video                         |
| 4    | `python ai_trainer/predictor.py --mode validate --full-pipeline` | Corta clips buenos/malos y actualiza dataset |

## Requisitos 
- Python 3.10
- ffmpeg (en PATH)
- Groq API key (opcional, para transcripción rápida)
