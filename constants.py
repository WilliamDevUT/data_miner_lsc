import os
import cv2
import json

# SETTINGS
MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 1662
MODEL_FRAMES = 15

# PATHS
ROOT_PATH = os.getcwd()
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATA_JSON_PATH = os.path.join(DATA_PATH, "data.json")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.keras")
KEYPOINTS_PATH = os.path.join(DATA_PATH, "keypoints")
WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, "words.json")

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

words_text = {
    "adios": "ADIÓS",
    "bien": "BIEN",
    "buenas_noches": "BUENAS NOCHES",
    "buenas_tardes": "BUENAS TARDES",
    "buenos_dias": "BUENOS DÍAS",
    "como_estas": "COMO ESTÁS",
    "disculpa": "DISCULPA",
    "gracias": "GRACIAS",
    "hola": "HOLA",
    "mal": "MAL",
    "mas_o_menos": "MAS O MENOS",
    "me_ayudas": "ME AYUDAS",
    "por_favor": "POR FAVOR",
}

# CARGAR word_ids DESDE EL JSON O USAR VALORES POR DEFECTO
def load_word_ids():
    """Carga los word_ids desde words.json o usa las claves de words_text"""
    try:
        if os.path.exists(WORDS_JSON_PATH):
            with open(WORDS_JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('word_ids', list(words_text.keys()))
        else:
            print(f"⚠ {WORDS_JSON_PATH} no encontrado, usando word_ids por defecto")
            return list(words_text.keys())
    except Exception as e:
        print(f"⚠ Error al cargar word_ids: {e}")
        return list(words_text.keys())

# Cargar word_ids automáticamente
word_ids = load_word_ids()

print(f"✓ Constantes cargadas: {len(word_ids)} palabras disponibles")