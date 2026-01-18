"""
Predictor de señas con sistema de rutas relativas y detección automática de frames.
Incluye validación contra Whisper y corte de clips de video.
Este script puede ejecutarse independientemente del data_miner.
"""
import json
import numpy as np
import os
import re
from datetime import datetime
from tensorflow import keras
from moviepy.editor import VideoFileClip
from pathlib import Path
import sys


# ======================================================
# CONFIGURACIÓN DE RUTAS PARA IMPORTAR CONFIG
# ======================================================
# Obtener el directorio del script actual (ai_trainer)
script_dir = Path(__file__).resolve().parent
print(f"📂 Directorio del script: {script_dir}")

# Subir un nivel para llegar al directorio raíz del proyecto (new)
project_root = script_dir.parent
print(f"📂 Directorio raíz del proyecto: {project_root}")

# Añadir el directorio raíz al path de Python
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Verificar que config.py existe
config_path = project_root / "config.py"
print(f"📂 Buscando config en: {config_path}")
print(f"📂 ¿Existe config.py?: {config_path.exists()}")

# Importar configuración compartida
try:
    from config import (
        get_model_paths, get_latest_video_dir, get_video_files,
        load_processing_state, save_processing_state,
        ensure_directories, validate_environment,
        PREDICTIONS_OUTPUT, PREDICTION_THRESHOLD, MODEL_FRAMES,
        VALIDATED_VIDEOS_DIR, VALIDATED_KEYPOINTS_DIR,
        UNKNOWN_VIDEOS_DIR, UNKNOWN_KEYPOINTS_DIR,
        get_whisper_and_video_paths 
    )
    CONFIG_AVAILABLE = True
    print("✓ Config compartido cargado exitosamente")
except ImportError as e:
    print(f"⚠ No se pudo importar config.py: {e}")
    print(f"⚠ Usando configuración por defecto")
    CONFIG_AVAILABLE = False
    PREDICTIONS_OUTPUT = Path("./output")
    PREDICTION_THRESHOLD = 0.7
    MODEL_FRAMES = 30

# Importar constantes del proyecto original (si están disponibles)
try:
    from constants import words_text
    print(f"✓ Constantes cargadas: {len(words_text)} palabras disponibles")
except ImportError:
    print("⚠ constants.py no disponible, usando diccionario básico")
    words_text = {}


# ======================================================
# CLASE: FileUtils
# ======================================================
class FileUtils:
    """Utilidades para manejo de archivos y carpetas."""
    
    @staticmethod
    def create_folder_if_not_exists(path):
        """Crea una carpeta si no existe."""
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  [INFO] Carpeta creada: {path}")
    
    @staticmethod
    def clean_filename(name):
        """
        Limpia un string para que sea un nombre de archivo válido.
        Convierte a minúsculas y elimina caracteres inválidos.
        """
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        # Eliminar espacios al inicio/final, convertir a minúsculas, eliminar comas finales
        name = name.strip().lower().rstrip(',')
        return name
    
    @staticmethod
    def get_next_video_number(folder_path):
        """
        Busca en la carpeta todos los archivos seña_N.mp4
        y devuelve el siguiente número disponible.
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            return 1
        
        files = list(folder_path.iterdir())
        existing_numbers = []
        
        # Patrón para buscar: seña_N.mp4
        pattern = re.compile(r'seña_(\d+)\.mp4')
        
        for file in files:
            match = pattern.match(file.name)
            if match:
                number = int(match.group(1))
                existing_numbers.append(number)
        
        return max(existing_numbers) + 1 if existing_numbers else 1


# ======================================================
# FUNCIÓN AUXILIAR: Normalizar Keypoints
# ======================================================
def normalize_keypoints(keypoints_sequence, target_frames):
    """
    Normaliza una secuencia de keypoints a un número fijo de frames.
    
    Args:
        keypoints_sequence: Lista de keypoints (frames variables)
        target_frames: Número de frames objetivo
    
    Returns:
        Array numpy normalizado con shape (target_frames, num_keypoints)
    """
    keypoints_array = np.array(keypoints_sequence)
    current_frames = len(keypoints_array)
    
    if current_frames == 0:
        num_keypoints = 1662  # Por defecto
        return np.zeros((target_frames, num_keypoints))
    
    if current_frames == target_frames:
        return keypoints_array
    
    elif current_frames > target_frames:
        indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
        return keypoints_array[indices]
    
    else:
        indices_original = np.arange(current_frames)
        indices_target = np.linspace(0, current_frames - 1, target_frames)
        
        normalized = np.zeros((target_frames, keypoints_array.shape[1]))
        for i in range(keypoints_array.shape[1]):
            normalized[:, i] = np.interp(indices_target, indices_original, keypoints_array[:, i])
        
        return normalized


# ======================================================
# FUNCIÓN AUXILIAR: Detectar frames del modelo
# ======================================================
def detect_model_frames(model):
    """
    Detecta el número de frames que espera el modelo.
    
    Args:
        model: Modelo Keras cargado
    
    Returns:
        int: Número de frames detectados
    """
    try:
        input_shape = model.input_shape
        if len(input_shape) > 1:
            return input_shape[1]
    except:
        pass
    
    return MODEL_FRAMES  # Valor por defecto del config


# ======================================================
# CLASE: WhisperTranscriptionLoader
# ======================================================
class WhisperTranscriptionLoader:
    """Cargador de transcripciones de Whisper desde JSON."""
    
    def __init__(self, json_path):
        """
        Inicializa el cargador de transcripciones.
        
        Args:
            json_path: Ruta al archivo JSON de Whisper
        """
        self.json_path = Path(json_path)
        self.words = []
        self._load_transcription()
    
    def _load_transcription(self):
        """Carga las transcripciones desde el JSON de Whisper."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.words = data.get('words', [])
            
            print(f"\n📝 Transcripción Whisper cargada")
            print(f"   Total de palabras: {len(self.words)}")
            
        except Exception as e:
            print(f"❌ Error al cargar transcripción Whisper: {e}")
            raise
    
    def get_word_at_time(self, time):
        """
        Obtiene la palabra que se estaba diciendo en un momento específico.
        
        Args:
            time: Tiempo en segundos
        
        Returns:
            Diccionario con la palabra o None si no hay coincidencia
        """
        for word_data in self.words:
            if word_data['start'] <= time <= word_data['end']:
                return word_data
        return None
    
    def get_words_in_range(self, start_time, end_time):
        """
        Obtiene todas las palabras en un rango de tiempo.
        
        Args:
            start_time: Tiempo de inicio en segundos
            end_time: Tiempo de fin en segundos
        
        Returns:
            Lista de palabras en el rango
        """
        words_in_range = []
        for word_data in self.words:
            if start_time <= word_data['start'] <= end_time:
                words_in_range.append(word_data)
        return words_in_range


# ======================================================
# CLASE: VideoClipper
# ======================================================
class VideoClipper:
    """Cortador de clips de video."""
    
    def __init__(self, video_path):
        """
        Inicializa el cortador de video.
        
        Args:
            video_path: Ruta al video fuente
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"No se encontró el video: {video_path}")
        
        # Obtener duración del video
        with VideoFileClip(str(self.video_path)) as video:
            self.video_duration = video.duration
        
        print(f"\n🎬 Video cargado: {self.video_path.name}")
        print(f"   Duración: {self.video_duration:.2f}s")
    
    def cut_clip(self, start_time, end_time, output_folder):
        """
        Corta un clip de video y lo guarda con numeración automática.
        
        Args:
            start_time: Tiempo de inicio en segundos
            end_time: Tiempo de fin en segundos
            output_folder: Carpeta donde guardar el clip
        
        Returns:
            str: Nombre del archivo guardado o None si falló
        """
        try:
            output_folder = Path(output_folder)
            
            # Obtener el siguiente número de video disponible
            video_number = FileUtils.get_next_video_number(output_folder)
            
            # Construir nombre del archivo: seña_N.mp4
            filename = f"seña_{video_number}.mp4"
            output_path = output_folder / filename
            
            # Ajustar tiempos si exceden la duración del video
            if start_time >= self.video_duration:
                print(f"  [ERROR] Tiempo de inicio ({start_time:.2f}s) excede la duración del video ({self.video_duration:.2f}s)")
                return None
            
            if end_time > self.video_duration:
                print(f"  [AVISO] Tiempo de fin ajustado de {end_time:.2f}s a {self.video_duration:.2f}s")
                end_time = self.video_duration
            
            # Cortar y guardar clip
            with VideoFileClip(str(self.video_path)) as video:
                clip = video.subclip(start_time, end_time)
                clip.write_videofile(
                    str(output_path), 
                    codec="libx264", 
                    audio_codec="aac", 
                    verbose=False, 
                    logger=None
                )
            
            print(f"  [✓ CLIP GUARDADO] {filename}")
            return filename
            
        except Exception as e:
            print(f"  [ERROR] No se pudo cortar el clip: {e}")
            return None


# ======================================================
# CLASE: SignPredictor
# ======================================================
class SignPredictor:
    """Clase para realizar predicciones de señas desde JSON de keypoints."""
    
    def __init__(self, model_path, words_json_path=None, threshold=None):
        """
        Inicializa el predictor.
        
        Args:
            model_path: Path al modelo Keras
            words_json_path: Path al words.json del modelo (opcional)
            threshold: Umbral de confianza (0.0 - 1.0)
        """
        self.model_path = Path(model_path)
        self.threshold = threshold or PREDICTION_THRESHOLD
        self.model = None
        self.all_detections = []
        self.model_frames = None  # Se detectará automáticamente
        
        # Cargar word_ids
        if words_json_path and Path(words_json_path).exists():
            self.words_json_path = Path(words_json_path)
            with open(words_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.word_ids = data.get('word_ids', [])
        else:
            self.words_json_path = None
            self.word_ids = []
            print("⚠ No se cargó words.json, usando índices numéricos")
        
        print(f"\n{'='*70}")
        print(f"🤖 INICIALIZANDO PREDICTOR DE SEÑAS")
        print(f"{'='*70}")
        print(f"Modelo: {self.model_path}")
        if self.words_json_path:
            print(f"Words JSON: {self.words_json_path}")
        print(f"Clases disponibles: {len(self.word_ids) if self.word_ids else 'N/A'}")
        print(f"Umbral: {self.threshold * 100}%")
        
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo Keras - VERSIÓN SIMPLIFICADA."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"No se encontró el modelo: {self.model_path}")
            
            print(f"\n📦 Cargando modelo...")
            self.model = keras.models.load_model(str(self.model_path))
            print(f"✓ Modelo cargado exitosamente")
            
            # Detectar frames del modelo
            self.model_frames = detect_model_frames(self.model)
            
            print(f"  • Input shape: {self.model.input_shape}")
            print(f"  • Output shape: {self.model.output_shape}")
            print(f"  • Frames detectados: {self.model_frames}")
            
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
            raise
    
    def predict_from_json(self, json_path, save_results=True, output_dir=None):
        """
        Procesa todos los segmentos del JSON y realiza predicciones.
        
        Args:
            json_path: Path al archivo JSON con keypoints
            save_results: Guardar resultados en JSON
            output_dir: Directorio para guardar resultados
        
        Returns:
            Lista de detecciones
        """
        json_path = Path(json_path)
        
        print(f"\n{'='*70}")
        print(f"🎯 PROCESANDO SEGMENTOS")
        print(f"{'='*70}")
        print(f"Archivo: {json_path}")
        print(f"Frames por predicción: {self.model_frames}")
        
        # Cargar JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                segments_data = json.load(f)
            print(f"✓ JSON cargado: {len(segments_data)} segmentos")
        except Exception as e:
            print(f"❌ Error al cargar JSON: {e}")
            return None
        
        # Reiniciar detecciones
        self.all_detections = []
        total_segments = len(segments_data)
        
        # Procesar cada segmento
        for idx, segment in enumerate(segments_data):
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', 0)
            keypoints_sequence = segment.get('keypoints', [])
            
            # Mostrar progreso
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"\n📍 Procesando segmento {idx + 1}/{total_segments}")
                print(f"   Tiempo: {start_time}s - {end_time}s")
                print(f"   Frames en segmento: {len(keypoints_sequence)}")
            
            if len(keypoints_sequence) == 0:
                if idx < 3:
                    print(f"   ⚠ Segmento vacío, saltando...")
                continue
            
            # Predicción
            try:
                # Normalizar a la cantidad exacta de frames que espera el modelo
                kp_normalized = normalize_keypoints(keypoints_sequence, self.model_frames)
                
                # Realizar predicción
                res = self.model.predict(np.expand_dims(kp_normalized, axis=0), verbose=0)[0]
                
                max_idx = np.argmax(res)
                confidence = res[max_idx] * 100
                
                if confidence > self.threshold * 100:
                    # Obtener palabra
                    if self.word_ids and max_idx < len(self.word_ids):
                        word_id = self.word_ids[max_idx].split('-')[0]
                        palabra_detectada = words_text.get(word_id, f"palabra_{max_idx}")
                    else:
                        palabra_detectada = f"clase_{max_idx}"
                    
                    detection = {
                        'segmento': idx + 1,
                        'tiempo_start': round(start_time, 2),
                        'tiempo_fin': round(end_time, 2),
                        'palabra_detectada': palabra_detectada,
                        'confianza': round(confidence, 2),
                        'frames_procesados': len(keypoints_sequence),
                        'frames_normalizados': self.model_frames,
                        'keypoints': keypoints_sequence  # Guardar keypoints para extraer después
                    }
                    self.all_detections.append(detection)
                    
                    print(f"   ✓ Detectado: '{palabra_detectada}' | Confianza: {confidence:.2f}%")
                else:
                    if idx < 3:
                        print(f"   ✗ Confianza baja: {confidence:.2f}%")
            
            except Exception as e:
                if idx < 3:
                    print(f"   ❌ Error en segmento {idx + 1}: {e}")
                continue
        
        # Resumen final
        print(f"\n{'='*70}")
        print(f"📊 RESUMEN DE PREDICCIONES")
        print(f"{'='*70}")
        print(f"Segmentos procesados: {total_segments}")
        print(f"Detecciones válidas: {len(self.all_detections)}")
        print(f"Tasa de detección: {(len(self.all_detections) / total_segments * 100):.2f}%")
        print(f"{'='*70}\n")
        
        # Mostrar detecciones
        if self.all_detections:
            print("🎤 PALABRAS DETECTADAS:")
            print("-" * 70)
            for det in self.all_detections:
                print(f"[{det['tiempo_start']}s - {det['tiempo_fin']}s] "
                      f"{det['palabra_detectada']} ({det['confianza']}%)")
            print("-" * 70)
        else:
            print("⚠ No se detectaron señas con suficiente confianza")
        
        # Guardar resultados
        if save_results and self.all_detections:
            output_dir = Path(output_dir) if output_dir else PREDICTIONS_OUTPUT
            self.save_results(output_dir)
        
        return self.all_detections
    
    def save_results(self, output_dir):
        """Guarda los resultados en JSON incluyendo los keypoints de cada detección."""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"detecciones_{timestamp}.json"
            
            results = {
                'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'modelo': self.model_path.name,
                'frames_modelo': self.model_frames,
                'umbral_confianza': self.threshold,
                'total_detecciones': len(self.all_detections),
                'detecciones': self.all_detections  # Ya incluye keypoints
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 Resultados guardados en: {output_file}")
            print(f"   ✓ Guardadas {len(self.all_detections)} detecciones con keypoints")
            return output_file
            
        except Exception as e:
            print(f"❌ Error al guardar resultados: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_transcript(self):
        """Genera un transcript ordenado de las palabras detectadas."""
        if not self.all_detections:
            return ""
        
        sorted_detections = sorted(self.all_detections, key=lambda x: x['tiempo_start'])
        words = [det['palabra_detectada'] for det in sorted_detections]
        return " ".join(words)


# ======================================================
# CLASE: SignValidatorAndClipper
# ======================================================
class SignValidatorAndClipper:
    """
    Validador de detecciones de señas y cortador de clips.
    Compara detecciones de señas con transcripciones de Whisper.
    """
    
    def __init__(self, 
                 detections_json_path,
                 whisper_json_path,
                 video_path,
                 validated_videos_dir,
                 validated_keypoints_dir,
                 unknown_videos_dir,
                 unknown_keypoints_dir,
                 search_window_seconds=10,
                 additional_seconds=5):
        """
        Inicializa el validador y cortador.
        
        Args:
            detections_json_path: Ruta al JSON de detecciones de señas
            whisper_json_path: Ruta al JSON de transcripción Whisper
            video_path: Ruta al video fuente
            validated_videos_dir: Carpeta para videos validados
            validated_keypoints_dir: Carpeta para keypoints validados
            unknown_videos_dir: Carpeta para videos desconocidos
            unknown_keypoints_dir: Carpeta para keypoints desconocidos
            search_window_seconds: Ventana de búsqueda hacia atrás (segundos)
            additional_seconds: Segundos adicionales después del fin para clips desconocidos
        """
        self.detections_json_path = Path(detections_json_path)
        self.validated_videos_dir = Path(validated_videos_dir)
        self.validated_keypoints_dir = Path(validated_keypoints_dir)
        self.unknown_videos_dir = Path(unknown_videos_dir)
        self.unknown_keypoints_dir = Path(unknown_keypoints_dir)
        self.search_window_seconds = search_window_seconds
        self.additional_seconds = additional_seconds
        
        # Crear carpetas principales
        FileUtils.create_folder_if_not_exists(validated_videos_dir)
        FileUtils.create_folder_if_not_exists(validated_keypoints_dir)
        FileUtils.create_folder_if_not_exists(unknown_videos_dir)
        FileUtils.create_folder_if_not_exists(unknown_keypoints_dir)
        
        # Cargar detecciones de señas
        self._load_detections()
        
        # Cargar transcripción de Whisper
        self.whisper_loader = WhisperTranscriptionLoader(whisper_json_path)
        
        # Inicializar cortador de video
        self.video_clipper = VideoClipper(video_path)
        
        # Contadores
        self.validated_count = 0
        self.failed_count = 0
        self.remaining_processed = 0
    
    def _load_detections(self):
        """Carga las detecciones desde el JSON."""
        try:
            with open(self.detections_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.detections = data.get('detecciones', [])
            
            print(f"\n🎯 Detecciones cargadas")
            print(f"   Total de detecciones: {len(self.detections)}")
            
        except Exception as e:
            print(f"❌ Error al cargar detecciones: {e}")
            raise
    
    def _normalize_text(self, text):
        """Normaliza texto para comparación (minúsculas, sin puntuación)."""
        return re.sub(r'[^\w\s]', '', text).lower().strip()
    
    def _save_keypoints(self, keypoints, word_folder, filename):
        """
        Guarda los keypoints en formato JSON.
        
        Args:
            keypoints: Lista de keypoints del clip
            word_folder: Carpeta de la palabra
            filename: Nombre del archivo (sin extensión)
        """
        try:
            keypoints_path = word_folder / f"{filename}.json"
            with open(keypoints_path, 'w', encoding='utf-8') as f:
                json.dump(keypoints, f, ensure_ascii=False, indent=2)
            print(f"  [✓ KEYPOINTS GUARDADOS] {filename}.json")
        except Exception as e:
            print(f"  [ERROR] No se pudieron guardar keypoints: {e}")
    
    def validate_and_clip_detections(self):
        """
        Procesa todas las detecciones: valida contra Whisper y corta clips.
        """
        print(f"\n{'='*80}")
        print(f"🔍 INICIANDO VALIDACIÓN Y CORTE DE CLIPS")
        print(f"{'='*80}")
        
        for idx, detection in enumerate(self.detections):
            detected_word = detection['palabra_detectada']
            start_time = detection['tiempo_start']
            end_time = detection['tiempo_fin']
            keypoints = detection.get('keypoints', [])
            
            print(f"\n[{idx + 1}/{len(self.detections)}] Procesando: '{detected_word}' ({start_time:.2f}s - {end_time:.2f}s)")
            print(f"  [DEBUG] Keypoints en detección: {len(keypoints)} frames")
            
            # Definir ventana de búsqueda (hacia atrás desde start_time)
            search_start = start_time - self.search_window_seconds
            search_end = start_time
            
            # Obtener palabras de Whisper en la ventana temporal
            whisper_words = self.whisper_loader.get_words_in_range(search_start, search_end)
            
            # Normalizar palabra detectada para comparación
            normalized_detected = self._normalize_text(detected_word)
            
            # Verificar si la palabra está en la transcripción
            is_validated = any(
                normalized_detected in self._normalize_text(w['word']) 
                for w in whisper_words
            )
            
            if is_validated:
                # PALABRA VALIDADA
                print(f"  [✓ VALIDADA] Encontrada en transcripción")
                self.validated_count += 1
                
                # Crear carpeta para la palabra validada
                folder_name = FileUtils.clean_filename(detected_word)
                video_word_folder = self.validated_videos_dir / folder_name
                keypoints_word_folder = self.validated_keypoints_dir / folder_name
                FileUtils.create_folder_if_not_exists(video_word_folder)
                FileUtils.create_folder_if_not_exists(keypoints_word_folder)
                
                # Cortar clip
                filename = self.video_clipper.cut_clip(start_time, end_time, video_word_folder)
                
                # Guardar keypoints DEL SEGMENTO ORIGINAL con el mismo nombre (sin extensión)
                if filename:
                    filename_base = filename.replace('.mp4', '')
                    if keypoints:
                        self._save_keypoints(keypoints, keypoints_word_folder, filename_base)
                        print(f"  [DEBUG] Guardados {len(keypoints)} frames de keypoints")
                    else:
                        print(f"  [AVISO] No hay keypoints para guardar")
                
            else:
                # PALABRA NO VALIDADA (DESCONOCIDA)
                print(f"  [✗ NO VALIDADA] No encontrada en rango [{search_start:.2f}s - {search_end:.2f}s]")
                self.failed_count += 1
                
                # Buscar qué palabra REAL estaba en ese momento en Whisper
                real_word_data = self.whisper_loader.get_word_at_time(start_time)
                
                if real_word_data:
                    real_word = real_word_data['word']
                    print(f"  [INFO] Palabra real en ese momento: '{real_word}'")
                    folder_name = FileUtils.clean_filename(real_word)
                else:
                    print(f"  [AVISO] No se encontró ninguna palabra en Whisper en el tiempo {start_time:.2f}s")
                    folder_name = "sin_palabra"
                
                # Crear carpeta con la palabra real
                video_word_folder = self.unknown_videos_dir / folder_name
                keypoints_word_folder = self.unknown_keypoints_dir / folder_name
                
                if video_word_folder.exists():
                    print(f"  [INFO] Carpeta '{folder_name}' ya existe, agregando clip...")
                else:
                    FileUtils.create_folder_if_not_exists(video_word_folder)
                    FileUtils.create_folder_if_not_exists(keypoints_word_folder)
                
                # Calcular tiempos del clip (desde end_time hasta end_time + additional_seconds)
                clip_start = end_time
                clip_end = end_time + self.additional_seconds
                
                print(f"  [INFO] Cortando clip desconocido: {clip_start:.2f}s - {clip_end:.2f}s")
                
                # Cortar clip
                filename = self.video_clipper.cut_clip(clip_start, clip_end, video_word_folder)
                
                # Guardar keypoints DEL SEGMENTO ORIGINAL (no del clip futuro)
                # Los keypoints corresponden al segmento detectado, no al clip desconocido
                if filename:
                    filename_base = filename.replace('.mp4', '')
                    if keypoints:
                        self._save_keypoints(keypoints, keypoints_word_folder, filename_base)
                        print(f"  [DEBUG] Guardados {len(keypoints)} frames de keypoints del segmento detectado")
                    else:
                        print(f"  [AVISO] No hay keypoints para guardar")
    
    def process_remaining_words(self):
        """
        Procesa las palabras restantes de Whisper después de la última detección.
        """
        print(f"\n{'='*80}")
        print(f"📝 PROCESANDO PALABRAS RESTANTES DEL WHISPER")
        print(f"{'='*80}")
        
        if not self.detections:
            print("\nNo había detecciones para procesar.")
            return
        
        # Encontrar el tiempo de la última detección
        last_detection_time = max(d['tiempo_fin'] for d in self.detections)
        print(f"\nÚltima detección terminó en: {last_detection_time:.2f}s")
        
        # Filtrar palabras de Whisper después de la última detección
        remaining_words = [
            w for w in self.whisper_loader.words 
            if w['start'] >= last_detection_time
        ]
        
        if not remaining_words:
            print("\nNo hay palabras restantes después de la última detección.")
            return
        
        print(f"Palabras restantes por procesar: {len(remaining_words)}")
        
        for idx, word_data in enumerate(remaining_words):
            real_word = word_data['word']
            word_start = word_data['start']
            word_end = word_data['end']
            
            print(f"\n[{idx + 1}/{len(remaining_words)}] Procesando palabra restante: '{real_word}' ({word_start:.2f}s - {word_end:.2f}s)")
            
            # Crear carpeta con la palabra real
            folder_name = FileUtils.clean_filename(real_word)
            video_word_folder = self.unknown_videos_dir / folder_name
            keypoints_word_folder = self.unknown_keypoints_dir / folder_name
            
            if video_word_folder.exists():
                print(f"  [INFO] Carpeta '{folder_name}' ya existe, agregando clip...")
            else:
                FileUtils.create_folder_if_not_exists(video_word_folder)
                FileUtils.create_folder_if_not_exists(keypoints_word_folder)
            
            # Calcular tiempos del clip
            clip_start = word_end
            clip_end = word_end + self.additional_seconds
            
            print(f"  [INFO] Cortando clip: {clip_start:.2f}s - {clip_end:.2f}s")
            
            # Cortar clip
            filename = self.video_clipper.cut_clip(clip_start, clip_end, video_word_folder)
            if filename:
                self.remaining_processed += 1
                # Guardar keypoints vacíos
                filename_base = filename.replace('.mp4', '')
                self._save_keypoints([], keypoints_word_folder, filename_base)
        
        print(f"\nPalabras restantes procesadas: {self.remaining_processed}")
    
    def print_summary(self):
        """Imprime un resumen final del proceso."""
        print(f"\n{'='*80}")
        print(f"✅ PROCESO COMPLETADO")
        print(f"{'='*80}")
        print(f"Total de detecciones procesadas: {len(self.detections)}")
        print(f"  - Validaciones exitosas: {self.validated_count}")
        print(f"  - Validaciones fallidas (desconocidas): {self.failed_count}")
        print(f"Palabras restantes del Whisper procesadas: {self.remaining_processed}")
        print(f"\nClips guardados en:")
        print(f"  - Validadas videos: {self.validated_videos_dir}")
        print(f"  - Validadas keypoints: {self.validated_keypoints_dir}")
        print(f"  - Desconocidas videos: {self.unknown_videos_dir}")
        print(f"  - Desconocidas keypoints: {self.unknown_keypoints_dir}")
        print(f"{'='*80}")


# ======================================================
# FUNCIONES DE ALTO NIVEL
# ======================================================
def predict_latest_video(model_path=None, words_json_path=None, threshold=None):
    """
    Procesa el video más reciente automáticamente.
    
    Args:
        model_path: Path al modelo (opcional, usa config si no se especifica)
        words_json_path: Path al words.json (opcional)
        threshold: Umbral de confianza (opcional)
    
    Returns:
        Lista de detecciones o None si hay error
    """
    if not CONFIG_AVAILABLE:
        print("❌ config.py no disponible, no se puede buscar video automáticamente")
        return None
    
    # Obtener rutas del modelo
    if model_path is None or words_json_path is None:
        default_model, default_words = get_model_paths()
        model_path = model_path or default_model
        words_json_path = words_json_path or default_words
    
    # Validar ambiente
    if not validate_environment(check_model=True, check_video=True):
        return None
    
    # Obtener último video
    latest_dir = get_latest_video_dir()
    if not latest_dir:
        print("❌ No se encontraron videos procesados")
        return None
    
    files = get_video_files(latest_dir)
    
    print(f"\n{'='*70}")
    print(f"🎬 VIDEO A PROCESAR")
    print(f"{'='*70}")
    print(f"Directorio: {latest_dir.name}")
    print(f"Keypoints: {files['keypoints'].name}")
    print(f"{'='*70}")
    
    # Crear predictor y procesar
    predictor = SignPredictor(model_path, words_json_path, threshold)
    return predictor.predict_from_json(files['keypoints'])


def predict_from_state():
    """
    Procesa el video indicado en el archivo de estado.
    Útil cuando data_miner acaba de terminar.
    """
    if not CONFIG_AVAILABLE:
        print("❌ config.py no disponible")
        return None
    
    state = load_processing_state()
    if not state:
        print("❌ No hay estado guardado")
        print("\n💡 Intenta usar --mode latest para procesar el video más reciente")
        return None
    
    keypoints_path = Path(state['files']['keypoints'])
    if not keypoints_path.exists():
        print(f"❌ Archivo de keypoints no existe: {keypoints_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f"📋 USANDO ESTADO GUARDADO")
    print(f"{'='*70}")
    print(f"Timestamp: {state['timestamp']}")
    print(f"Video: {state['video_dir_name']}")
    print(f"{'='*70}")
    
    model_path, words_path = get_model_paths()
    predictor = SignPredictor(model_path, words_path)
    return predictor.predict_from_json(keypoints_path)


def validate_and_clip_pipeline(detections_json, whisper_json, video_path, 
                                validated_videos_dir, validated_keypoints_dir,
                                unknown_videos_dir, unknown_keypoints_dir,
                                search_window=10, additional_seconds=5):
    """
    Pipeline completo de validación y corte de clips.
    
    Args:
        detections_json: JSON con detecciones de señas
        whisper_json: JSON con transcripción de Whisper
        video_path: Path al video fuente
        validated_videos_dir: Carpeta para videos validados
        validated_keypoints_dir: Carpeta para keypoints validados
        unknown_videos_dir: Carpeta para videos desconocidos
        unknown_keypoints_dir: Carpeta para keypoints desconocidos
        search_window: Ventana de búsqueda hacia atrás (segundos)
        additional_seconds: Segundos adicionales después del fin
    
    Returns:
        SignValidatorAndClipper instance con resultados
    """
    validator = SignValidatorAndClipper(
        detections_json_path=detections_json,
        whisper_json_path=whisper_json,
        video_path=video_path,
        validated_videos_dir=validated_videos_dir,
        validated_keypoints_dir=validated_keypoints_dir,
        unknown_videos_dir=unknown_videos_dir,
        unknown_keypoints_dir=unknown_keypoints_dir,
        search_window_seconds=search_window,
        additional_seconds=additional_seconds
    )
    
    # Validar y cortar clips de detecciones
    validator.validate_and_clip_detections()
    
    # Procesar palabras restantes del Whisper
    validator.process_remaining_words()
    
    # Imprimir resumen final
    validator.print_summary()
    
    return validator


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predictor de señas LSM con validación y corte de clips')
    
    # Modos de operación
    parser.add_argument('--mode', choices=['latest', 'state', 'manual', 'validate'], default='latest',
                       help='Modo de operación')
    
    # Parámetros para predicción
    parser.add_argument('--keypoints', type=str, help='Ruta manual al JSON de keypoints')
    parser.add_argument('--model', type=str, help='Ruta manual al modelo')
    parser.add_argument('--words', type=str, help='Ruta manual al words.json')
    parser.add_argument('--threshold', type=float, help='Umbral de confianza (0.0-1.0)')
    
    # Parámetros para validación
    parser.add_argument('--detections', type=str, help='JSON con detecciones de señas')
    parser.add_argument('--whisper', type=str, help='JSON con transcripción Whisper')
    parser.add_argument('--video', type=str, help='Ruta al video fuente')
    parser.add_argument('--search-window', type=int, default=10, help='Ventana de búsqueda (segundos)')
    parser.add_argument('--additional-seconds', type=int, default=5, help='Segundos adicionales en clips')
    
    # Modo completo (predicción + validación)
    parser.add_argument('--full-pipeline', action='store_true', help='Ejecutar pipeline completo')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 PREDICTOR DE SEÑAS - VERSIÓN SIMPLIFICADA")
    print("="*70)
    
    if CONFIG_AVAILABLE:
        ensure_directories()
    
    try:
        # ===== MODO VALIDACIÓN =====
        if args.mode == 'validate':
            print("\n📍 Modo: Validación y corte de clips")
            
            # Paso 2: Validación - obtener rutas automáticamente si no se especificaron
            whisper_json = args.whisper
            video_path = args.video

            if not whisper_json or not video_path:
                # Intentar obtener del último video procesado
                auto_whisper, auto_video = get_whisper_and_video_paths()
                whisper_json = whisper_json or auto_whisper
                video_path = video_path or auto_video
                
                if not whisper_json or not video_path:
                    print("\n⚠ No se pudo encontrar automáticamente whisper/video.")
                    print("   Especifica --whisper y --video manualmente.")
                    print("💾 Detecciones guardadas. Usa --mode validate para continuar.")
                    sys.exit(0)
                
                print(f"\n✓ Usando automáticamente:")
                print(f"  Whisper: {whisper_json}")
                print(f"  Video: {video_path}")
            
            validate_and_clip_pipeline(
                detections_json=args.detections,
                whisper_json=whisper_json,
                video_path=video_path,
                validated_videos_dir=VALIDATED_VIDEOS_DIR,
                validated_keypoints_dir=VALIDATED_KEYPOINTS_DIR,
                unknown_videos_dir=UNKNOWN_VIDEOS_DIR,
                unknown_keypoints_dir=UNKNOWN_KEYPOINTS_DIR,
                search_window=args.search_window,
                additional_seconds=args.additional_seconds
            )
        
        # ===== MODO PIPELINE COMPLETO =====
        elif args.full_pipeline:
            print("\n📍 Modo: Pipeline completo (predicción + validación)")
            
            # Paso 1: Predicción
            if args.mode == 'latest':
                detections = predict_latest_video(
                    model_path=args.model,
                    words_json_path=args.words,
                    threshold=args.threshold
                )
            elif args.mode == 'state':
                detections = predict_from_state()
            elif args.mode == 'manual':
                if not args.keypoints:
                    print("❌ Debe especificar --keypoints en modo manual")
                    sys.exit(1)
                
                model_path = args.model
                words_path = args.words
                
                if not model_path and CONFIG_AVAILABLE:
                    model_path, words_path = get_model_paths()
                elif not model_path:
                    print("❌ Debe especificar --model en modo manual sin config.py")
                    sys.exit(1)
                
                predictor = SignPredictor(model_path, words_path, args.threshold)
                detections = predictor.predict_from_json(args.keypoints)
            
            if not detections:
                print("❌ No se generaron detecciones. Abortando pipeline.")
                sys.exit(1)
            
            # Paso 2: Validación - obtener rutas automáticamente
            whisper_json = args.whisper
            video_path = args.video
            
            if not whisper_json or not video_path:
                auto_whisper, auto_video = get_whisper_and_video_paths()
                whisper_json = whisper_json or auto_whisper
                video_path = video_path or auto_video
                
                if not whisper_json or not video_path:
                    print("\n⚠ No se pudo encontrar automáticamente whisper/video.")
                    print("   Especifica --whisper y --video manualmente.")
                    print("💾 Detecciones guardadas. Usa --mode validate para continuar.")
                    sys.exit(0)
                
                print(f"\n✓ Usando automáticamente:")
                print(f"  Whisper: {whisper_json}")
                print(f"  Video: {video_path}")
            
            # Obtener ruta del JSON de detecciones guardado
            output_dir = PREDICTIONS_OUTPUT if CONFIG_AVAILABLE else Path("./output")
            detection_files = sorted(output_dir.glob("detecciones_*.json"))
            if detection_files:
                detections_json = detection_files[-1]
                
                validate_and_clip_pipeline(
                    detections_json=detections_json,
                    whisper_json=whisper_json,
                    video_path=video_path,
                    validated_videos_dir=VALIDATED_VIDEOS_DIR,
                    validated_keypoints_dir=VALIDATED_KEYPOINTS_DIR,
                    unknown_videos_dir=UNKNOWN_VIDEOS_DIR,
                    unknown_keypoints_dir=UNKNOWN_KEYPOINTS_DIR,
                    search_window=args.search_window,
                    additional_seconds=args.additional_seconds
                )
            else:
                print("❌ No se encontró el archivo de detecciones guardado.")
                sys.exit(1)
        
        # ===== MODOS DE SOLO PREDICCIÓN =====
        else:
            if args.mode == 'latest':
                # Procesar video más reciente
                print("\n📍 Modo: Procesar video más reciente")
                detections = predict_latest_video(
                    model_path=args.model,
                    words_json_path=args.words,
                    threshold=args.threshold
                )
            
            elif args.mode == 'state':
                # Procesar según archivo de estado
                print("\n📍 Modo: Usar estado guardado")
                detections = predict_from_state()
            
            elif args.mode == 'manual':
                # Modo manual
                print("\n📍 Modo: Manual")
                if not args.keypoints:
                    print("❌ Debe especificar --keypoints en modo manual")
                    sys.exit(1)
                
                model_path = args.model
                words_path = args.words
                
                if not model_path and CONFIG_AVAILABLE:
                    model_path, words_path = get_model_paths()
                elif not model_path:
                    print("❌ Debe especificar --model en modo manual sin config.py")
                    sys.exit(1)
                
                predictor = SignPredictor(model_path, words_path, args.threshold)
                detections = predictor.predict_from_json(args.keypoints)
            
            # Mostrar transcript final
            if detections:
                predictor = SignPredictor.__new__(SignPredictor)
                predictor.all_detections = detections
                transcript = predictor.get_transcript()
                
                print(f"\n{'='*70}")
                print(f"📝 TRANSCRIPT COMPLETO")
                print(f"{'='*70}")
                print(transcript)
                print(f"{'='*70}\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Proceso interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)