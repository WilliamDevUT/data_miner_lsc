# helpers.py - VERSIÓN ACTUALIZADA PARA TU PROYECTO

import json
import os
import cv2
import numpy as np
import pandas as pd
from typing import NamedTuple
from constants import *

# GENERAL
def mediapipe_detection(image, model):
    """
    Convierte la imagen a RGB y la procesa con el modelo de MediaPipe.
    Devuelve los resultados de la detección.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return results

def create_folder(path):
    '''
    ### CREAR CARPETA SI NO EXISTE
    Si ya existe, no hace nada.
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def there_hand(results: NamedTuple) -> bool:
    """Verifica si se detectaron landmarks de mano izquierda o derecha."""
    return results.left_hand_landmarks or results.right_hand_landmarks

def get_word_ids(path):
    """Obtiene los IDs de las palabras desde un archivo JSON."""
    with open(path, 'r') as json_file:
        data = json.load(json_file)
        return data.get('word_ids')

def save_frames(frames, output_folder):
    """Guarda una lista de frames como imágenes JPG en una carpeta."""
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
        cv2.imwrite(frame_path, frame)

# CREATE KEYPOINTS
def extract_keypoints(results):
    """Extrae y aplana los keypoints de pose, cara y manos a partir de los resultados de MediaPipe."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def get_keypoints(model, sample_path):
    '''
    ### OBTENER KEYPOINTS DE LA MUESTRA
    Retorna la secuencia de keypoints de la muestra.
    '''
    kp_seq = []
    image_files = sorted(os.listdir(sample_path), key=lambda x: int(os.path.splitext(x)[0]))
    for img_name in image_files:
        img_path = os.path.join(sample_path, img_name)
        frame = cv2.imread(img_path)
        results = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)
        kp_seq.append(kp_frame)
    return np.array(kp_seq)

def insert_keypoints_sequence(df, n_sample:int, kp_seq):
    '''
    ### INSERTA LOS KEYPOINTS DE LA MUESTRA AL DATAFRAME
    Retorna el mismo DataFrame pero con los keypoints de la muestra agregados.
    '''
    for frame, keypoints in enumerate(kp_seq):
        data = {'sample': n_sample, 'frame': frame + 1, 'keypoints': [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints], ignore_index=True)
    
    return df

# TRAINING MODEL
def get_sequences_and_labels(words_id):
    """
    Carga secuencias de keypoints y sus correspondientes etiquetas (labels)
    desde archivos HDF5.
    
    Args:
        words_id: Lista de identificadores de palabras
    
    Returns:
        tuple: (sequences, labels)
            - sequences: Lista de secuencias de keypoints
            - labels: Lista de índices correspondientes a cada palabra
    """
    sequences, labels = [], []
    
    for word_index, word_id in enumerate(words_id):
        hdf_path = os.path.join(KEYPOINTS_PATH, f"{word_id}.h5")
        if not os.path.exists(hdf_path):
            print(f"⚠ Advertencia: No se encontró el archivo de keypoints {hdf_path}. Se omitirá.")
            continue
        
        try:
            data = pd.read_hdf(hdf_path, key='data')
            # Agrupa por el número de muestra
            for _, df_sample in data.groupby('sample'):
                # Convierte la columna de keypoints de la muestra en una lista
                seq_keypoints = np.array(df_sample['keypoints'].tolist())
                sequences.append(seq_keypoints)
                labels.append(word_index)
        except Exception as e:
            print(f"⚠ Error al cargar {hdf_path}: {e}")
            continue
                    
    return sequences, labels

def normalize_keypoints(keypoints_sequence, target_frames):
    """
    Normaliza una secuencia de keypoints a un número fijo de frames.
    
    Esta función maneja tres casos:
    1. Si la secuencia tiene exactamente el número de frames objetivo: retorna sin modificar
    2. Si tiene más frames: hace submuestreo tomando frames uniformemente distribuidos
    3. Si tiene menos frames: hace sobremuestreo interpolando linealmente
    
    Args:
        keypoints_sequence: Lista o array de keypoints con shape (frames_variables, num_keypoints)
        target_frames: Número de frames objetivo (típicamente MODEL_FRAMES)
    
    Returns:
        Array numpy normalizado con shape (target_frames, num_keypoints)
    """
    # Convertir a array numpy si no lo es
    keypoints_array = np.array(keypoints_sequence)
    current_frames = len(keypoints_array)
    
    # Caso especial: secuencia vacía
    if current_frames == 0:
        return np.zeros((target_frames, LENGTH_KEYPOINTS))
    
    # Caso 1: Ya tiene el tamaño correcto
    if current_frames == target_frames:
        return keypoints_array
    
    # Caso 2: Tiene más frames que el objetivo (submuestreo)
    elif current_frames > target_frames:
        # Seleccionar frames uniformemente distribuidos
        indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
        return keypoints_array[indices]
    
    # Caso 3: Tiene menos frames que el objetivo (sobremuestreo/interpolación)
    else:
        # Crear índices para interpolación
        indices_original = np.arange(current_frames)
        indices_target = np.linspace(0, current_frames - 1, target_frames)
        
        # Interpolar cada dimensión de keypoints
        normalized = np.zeros((target_frames, keypoints_array.shape[1]))
        
        for i in range(keypoints_array.shape[1]):
            normalized[:, i] = np.interp(
                indices_target,
                indices_original,
                keypoints_array[:, i]
            )
        
        return normalized


# EJEMPLOS DE USO Y PRUEBAS
if __name__ == "__main__":
    print("=== PRUEBAS DE FUNCIONES HELPER ===\n")
    
    # Prueba 1: get_word_ids
    print("Prueba 1: Cargar word_ids desde JSON")
    if os.path.exists(WORDS_JSON_PATH):
        word_ids = get_word_ids(WORDS_JSON_PATH)
        print(f"✓ Word IDs cargados: {word_ids}\n")
    else:
        print(f"⚠ Archivo {WORDS_JSON_PATH} no encontrado\n")
    
    # Prueba 2: normalize_keypoints
    print("Prueba 2: Normalización de keypoints")
    
    # Secuencia corta (10 frames → 15 frames)
    kp_short = np.random.rand(10, LENGTH_KEYPOINTS)
    normalized_long = normalize_keypoints(kp_short, MODEL_FRAMES)
    print(f"  10 frames → {MODEL_FRAMES} frames: {kp_short.shape} → {normalized_long.shape}")
    
    # Secuencia larga (30 frames → 15 frames)
    kp_long = np.random.rand(30, LENGTH_KEYPOINTS)
    normalized_short = normalize_keypoints(kp_long, MODEL_FRAMES)
    print(f"  30 frames → {MODEL_FRAMES} frames: {kp_long.shape} → {normalized_short.shape}")
    
    # Secuencia exacta (15 frames → 15 frames)
    kp_exact = np.random.rand(MODEL_FRAMES, LENGTH_KEYPOINTS)
    normalized_same = normalize_keypoints(kp_exact, MODEL_FRAMES)
    print(f"  {MODEL_FRAMES} frames → {MODEL_FRAMES} frames: {kp_exact.shape} → {normalized_same.shape}")
    print(f"  ¿Sin cambios? {np.array_equal(kp_exact, normalized_same)}\n")
    
    # Prueba 3: get_sequences_and_labels (si existen archivos HDF5)
    print("Prueba 3: Cargar secuencias y etiquetas")
    if os.path.exists(KEYPOINTS_PATH):
        h5_files = [f for f in os.listdir(KEYPOINTS_PATH) if f.endswith('.h5')]
        if len(h5_files) > 0:
            print(f"  Archivos HDF5 encontrados: {len(h5_files)}")
            # Cargar solo el primero como prueba
            test_word = h5_files[0].replace('.h5', '')
            sequences, labels = get_sequences_and_labels([test_word])
            print(f"  ✓ Cargadas {len(sequences)} secuencias de '{test_word}'")
            if len(sequences) > 0:
                print(f"    Shape primera secuencia: {sequences[0].shape}")
        else:
            print("  ⚠ No se encontraron archivos HDF5")
    else:
        print(f"  ⚠ Carpeta {KEYPOINTS_PATH} no encontrada")