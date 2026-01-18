"""
Data Miner - Procesador de videos con sistema de rutas relativas.
Compatible con el sistema de configuración compartida.
"""
import os
import yt_dlp
from pydub import AudioSegment
from dotenv import load_dotenv
import json
from moviepy.editor import VideoFileClip
from datetime import datetime
import numpy as np
import cv2
from mediapipe.python.solutions.holistic import Holistic
from pathlib import Path
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))

# Subir un nivel en la estructura de directorios para llegar a la carpeta 'new'
parent_dir = os.path.dirname(script_dir)

# Añadir el directorio padre a la lista de rutas donde Python busca módulos
sys.path.append(parent_dir)
# Importar configuración compartida


# Intentar importar configuración compartida
try:
    from config import (
        DATA_MINER_OUTPUT, SEGMENT_DURATION, save_processing_state,
        ensure_directories
    )
    CONFIG_AVAILABLE = True
    print("✓ Config compartido cargado")
except ImportError:
    print("⚠ Config no disponible, usando valores por defecto")
    CONFIG_AVAILABLE = False
    DATA_MINER_OUTPUT = Path("./vid_data")
    SEGMENT_DURATION = 2

# Importar helpers y constants del proyecto
try:
    from helpers import *
    from constants import *
    print("✓ Helpers y constants importados")
except ImportError as e:
    print(f"⚠ No se pudieron importar helpers/constants: {e}")
    # Definir funciones básicas si no están disponibles
    def mediapipe_detection(image, model):
        """Fallback básico para detección."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results
    
    def extract_keypoints(results):
        """Fallback básico para extracción de keypoints."""
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in 
                        results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in 
                        results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in 
                      results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in 
                      results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    
    def draw_keypoints(image, results):
        """Fallback básico para dibujar keypoints."""
        pass

# Importar whisper si está disponible
try:
    from whisper import transcribe_audio_in_chunks
    WHISPER_AVAILABLE = True
except ImportError:
    print("⚠ Whisper no disponible")
    WHISPER_AVAILABLE = False

load_dotenv()


# ======================================================
# 1. Segmentador de Signs (basado en tiempo)
# ======================================================
class SignSegments:
    """Divide videos en segmentos de tiempo fijo."""
    
    def segment(self, video_path, times_between=None):
        """
        Divide el video en segmentos de tiempo específico.
        
        Args:
            video_path: Path al archivo de video
            times_between: Segundos entre cada segmento (usa SEGMENT_DURATION si es None)
        
        Returns:
            Lista de diccionarios con start_time y end_time
        """
        if times_between is None:
            times_between = SEGMENT_DURATION
        
        video_path = Path(video_path)
        
        print(f"--- Analizando '{video_path.name}' para obtener su duración... ---")
        
        if not video_path.exists():
            print(f"❌ Error: El archivo de video no existe: {video_path}")
            return None

        try:
            clip = VideoFileClip(str(video_path))
            total_duration = clip.duration
            clip.close()
            
            number_of_parts = int(total_duration / times_between)
            print(f"✓ Duración total: {total_duration:.2f}s")
            print(f"✓ Dividiendo en {number_of_parts} segmentos de {times_between}s cada uno")
            
            segments_list = []
            current_start = 0.0
            
            for i in range(number_of_parts):
                current_end = current_start + times_between
                
                segment = {
                    "start_time": round(current_start, 2),
                    "end_time": round(current_end, 2)
                }
                segments_list.append(segment)
                
                if (i + 1) % 10 == 0:
                    print(f"  Generados {i + 1}/{number_of_parts} segmentos...")
                
                current_start = current_end
            
            print(f"✓ {len(segments_list)} segmentos generados")
            return segments_list

        except Exception as e:
            print(f"❌ Error al procesar el video: {e}")
            return None


# ======================================================
# 2. Clase principal de procesamiento
# ======================================================
class VideoTask:
    """Gestiona el procesamiento completo de videos."""
    
    def __init__(self, url=None, base_dir=None, output_dir=None, video_path=None):
        """
        Inicializa la tarea de procesamiento de video.
        
        Args:
            url: URL del video a descargar (opcional)
            base_dir: Directorio base para archivos (usa auto si es None)
            output_dir: Directorio de salida (obsoleto, se usa base_dir)
            video_path: Ruta a video local existente (opcional)
        """
        # Si no se especifica base_dir, crear uno con timestamp
        if base_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = DATA_MINER_OUTPUT / f"video_{timestamp}"
        else:
            base_dir = Path(base_dir)
        
        self.url = url
        self.base_dir = base_dir
        self.video_path_local = Path(video_path) if video_path else None
        
        # Rutas de archivos usando Path
        self.video_file = self.base_dir / "video.mp4"
        self.audio_file = self.base_dir / "audio.wav"
        self.keypoints_file = self.base_dir / "sign.json"
        self.intervals_file = self.base_dir / "signInterval.json"

        # Crear directorios
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"📁 CONFIGURACIÓN DE TAREA")
        print(f"{'='*70}")
        print(f"Directorio de trabajo: {self.base_dir}")
        if url:
            print(f"URL: {url}")
        if video_path:
            print(f"Video local: {video_path}")
        print(f"{'='*70}\n")

    # ----------------------------------------------
    # Descarga de video
    # ----------------------------------------------
    def download(self):
        """Descarga el video desde la URL especificada."""
        if not self.url:
            print("⚠ No se proporcionó URL, saltando descarga")
            return

        print(f"📥 Descargando video desde: {self.url}")
        
        opts = {
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": str(self.video_file),
            "quiet": False
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([self.url])
            print("✓ Video descargado exitosamente")
        except Exception as e:
            print(f"❌ Error al descargar: {e}")
            raise

    # ----------------------------------------------
    # Copiar video local
    # ----------------------------------------------
    def copy_local_video(self):
        """Copia un video local al directorio de trabajo."""
        if not self.video_path_local:
            return
        
        if not self.video_path_local.exists():
            raise FileNotFoundError(f"Video no encontrado: {self.video_path_local}")
        
        print(f"📋 Copiando video local...")
        
        try:
            import shutil
            shutil.copy2(self.video_path_local, self.video_file)
            print(f"✓ Video copiado: {self.video_file.name}")
        except Exception as e:
            print(f"❌ Error al copiar video: {e}")
            raise

    # ----------------------------------------------
    # Extracción de audio
    # ----------------------------------------------
    def extract_audio(self):
        """Extrae el audio del video en formato WAV."""
        print("🎵 Extrayendo audio del video...")
        
        if not self.video_file.exists():
            raise FileNotFoundError(f"Video no encontrado: {self.video_file}")
        
        try:
            audio = AudioSegment.from_file(str(self.video_file))
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(str(self.audio_file), format="wav")
            print("✓ Audio extraído exitosamente")
        except Exception as e:
            print(f"❌ Error al extraer audio: {e}")
            raise

    # ----------------------------------------------
    # Transcripción
    # ----------------------------------------------
    def transcribe(self):
        """Transcribe el audio usando Whisper (si está disponible)."""
        if not WHISPER_AVAILABLE:
            print("⚠ Whisper no disponible, saltando transcripción")
            return
        
        print("🎤 Transcribiendo audio...")
        try:
            transcribe_audio_in_chunks(self.audio_file, self.base_dir)
            print("✓ Transcripción completada")
        except Exception as e:
            print(f"⚠ Error en transcripción: {e}")

    # ----------------------------------------------
    # Guardado de JSON
    # ----------------------------------------------
    def save_data_to_json(self, json_path, data):
        """
        Guarda datos en formato JSON de manera segura.
        
        Args:
            json_path: Path del archivo JSON
            data: Datos a guardar
        """
        json_path = Path(json_path)
        
        try:
            # Eliminar archivo anterior si existe
            if json_path.exists():
                json_path.unlink()
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, separators=(',', ': '))
            
            # Verificar guardado
            with open(json_path, 'r', encoding='utf-8') as f:
                verification = json.load(f)
            
            if isinstance(verification, list) and len(verification) > 0:
                print(f"✓ Datos guardados: {json_path.name}")
                print(f"  • {len(verification)} elementos guardados")
            else:
                print(f"⚠ Advertencia: Estructura inesperada en {json_path.name}")
                
        except Exception as e:
            print(f"❌ Error al guardar JSON: {e}")
            import traceback
            traceback.print_exc()
            raise

    # ----------------------------------------------
    # Extracción de keypoints por segmentos
    # ----------------------------------------------
    def extraer_keypoints_por_segmentos(self, video_path, json_path, 
                                       output_json_path=None, show_visualization=False):
        """
        Extrae keypoints de MediaPipe de cada frame en los intervalos especificados.
        
        Args:
            video_path: Path al video
            json_path: Path al JSON con intervalos
            output_json_path: Path de salida (opcional)
            show_visualization: Mostrar ventana de visualización
        
        Returns:
            Path al archivo JSON generado
        """
        video_path = Path(video_path)
        json_path = Path(json_path)
        
        # Preparar ruta de salida
        if output_json_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_json_path = self.base_dir / f"keypoints_{timestamp}.json"
        else:
            output_json_path = Path(output_json_path)
        
        # Cargar intervalos
        try:
            with open(json_path, 'r') as f:
                intervals = json.load(f)
            print(f"\n{'='*60}")
            print(f"🎯 EXTRACCIÓN DE KEYPOINTS")
            print(f"{'='*60}")
            print(f"✓ {len(intervals)} intervalos cargados")
        except FileNotFoundError:
            print(f"❌ No se encontró: {json_path}")
            return None
        except json.JSONDecodeError:
            print(f"❌ JSON corrupto: {json_path}")
            return None

        # Abrir video
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            print(f"❌ No se pudo abrir: {video_path}")
            return None
        
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames_video = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"✓ Video: {fps:.2f} FPS, {total_frames_video} frames totales")
        
        # Procesar con MediaPipe
        all_segments = []
        
        print(f"\n🔄 Procesando intervalos...")
        
        with Holistic() as holistic_model:
            for idx, interval in enumerate(intervals):
                start_time = float(interval['start_time'])
                end_time = float(interval['end_time'])
                
                # Mostrar progreso cada 10 segmentos
                if (idx + 1) % 10 == 0:
                    print(f"  [{idx + 1}/{len(intervals)}] Procesando...")
                
                # Posicionar video
                video.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
                
                # Extraer keypoints de cada frame
                frames_keypoints = []
                
                while True:
                    current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    if current_time > end_time:
                        break
                    
                    ret, frame = video.read()
                    if not ret:
                        break
                    
                    # Detectar y extraer keypoints
                    results = mediapipe_detection(frame, holistic_model)
                    kp_array = extract_keypoints(results)
                    
                    # Convertir a lista
                    if isinstance(kp_array, np.ndarray):
                        kp_list = kp_array.tolist()
                    else:
                        kp_list = list(kp_array)
                    
                    frames_keypoints.append(kp_list)
                    
                    # Visualización opcional
                    if show_visualization:
                        draw_keypoints(frame, results)
                        cv2.putText(frame, f"Segmento {idx + 1}/{len(intervals)}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow('Keypoints', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\n⚠ Interrumpido por usuario")
                            video.release()
                            cv2.destroyAllWindows()
                            return None
                
                # Guardar segmento
                if len(frames_keypoints) > 0:
                    segment = {
                        "start_time": round(start_time, 2),
                        "end_time": round(end_time, 2),
                        "keypoints": frames_keypoints
                    }
                    all_segments.append(segment)
        
        # Cerrar recursos
        video.release()
        cv2.destroyAllWindows()
        
        # Resumen
        if all_segments:
            total_frames = sum(len(seg['keypoints']) for seg in all_segments)
            
            print(f"\n{'='*60}")
            print(f"✓ EXTRACCIÓN COMPLETADA")
            print(f"{'='*60}")
            print(f"  • Segmentos procesados: {len(all_segments)}")
            print(f"  • Total frames: {total_frames}")
            print(f"  • Keypoints por frame: {len(all_segments[0]['keypoints'][0]) if all_segments else 0}")
            print(f"{'='*60}\n")
            
            # Guardar
            self.save_data_to_json(output_json_path, all_segments)
            return output_json_path
        else:
            print("\n❌ No se extrajo ningún dato")
            return None

    # ----------------------------------------------
    # Guardar intervalos de signos
    # ----------------------------------------------
    def save_sign_intervals(self, intervals):
        """Guarda los intervalos de tiempo en JSON."""
        if intervals is None:
            print("⚠ No hay intervalos para guardar")
            return
        
        print(f"💾 Guardando {len(intervals)} intervalos...")
        self.save_data_to_json(self.intervals_file, intervals)

    # ----------------------------------------------
    # Segmentar signos
    # ----------------------------------------------
    def segment_signs(self, sign_segmenter: SignSegments, times_between=None):
        """Divide el video en segmentos de tiempo."""
        return sign_segmenter.segment(self.video_file, times_between)

    # ----------------------------------------------
    # Proceso completo
    # ----------------------------------------------
    def process(self, sign_segmenter: SignSegments, times_between=None, 
                visualize=False, skip_transcription=False):
        """
        Ejecuta el pipeline completo de procesamiento.
        
        Args:
            sign_segmenter: Instancia de SignSegments
            times_between: Segundos entre cada segmento (usa SEGMENT_DURATION si es None)
            visualize: Mostrar visualización durante extracción
            skip_transcription: Saltar transcripción de audio
        """
        if times_between is None:
            times_between = SEGMENT_DURATION
        
        print("\n" + "="*70)
        print("🚀 INICIANDO PROCESAMIENTO DE VIDEO")
        print("="*70 + "\n")
        
        metadata = {
            'segment_duration': times_between,
            'visualization': visualize,
            'transcription': not skip_transcription
        }
        
        try:
            # 1. Obtener video
            if self.url:
                print("📥 PASO 1: Descarga desde URL")
                self.download()
            elif self.video_path_local:
                print("📋 PASO 1: Copiando video local")
                self.copy_local_video()
            else:
                print("⚠ PASO 1: Usando video existente")
                if not self.video_file.exists():
                    raise FileNotFoundError(f"Video no encontrado: {self.video_file}")
            
            # 2. Extraer audio
            print("\n🎵 PASO 2: Extracción de audio")
            self.extract_audio()
            
            # 3. Transcribir (opcional)
            if not skip_transcription:
                print("\n🎤 PASO 3: Transcripción")
                try:
                    self.transcribe()
                except Exception as e:
                    print(f"⚠ Transcripción omitida: {e}")
            else:
                print("\n⏭️  PASO 3: Transcripción omitida")
            
            # 4. Segmentar intervalos
            print(f"\n✂️  PASO 4: Segmentación de intervalos ({times_between}s cada uno)")
            intervals = self.segment_signs(sign_segmenter, times_between)
            
            if intervals is None:
                raise Exception("No se pudieron generar intervalos")
            
            metadata['total_segments'] = len(intervals)
            
            # 5. Guardar intervalos
            print("\n💾 PASO 5: Guardando intervalos")
            self.save_sign_intervals(intervals)
            
            # 6. Extraer keypoints
            print("\n🎯 PASO 6: Extracción de keypoints")
            keypoints_result = self.extraer_keypoints_por_segmentos(
                video_path=self.video_file,
                json_path=self.intervals_file,
                output_json_path=self.keypoints_file,
                show_visualization=visualize
            )
            
            if keypoints_result is None:
                raise Exception("Falló la extracción de keypoints")
            
            # 7. Guardar estado compartido (si config está disponible)
            if CONFIG_AVAILABLE:
                print("\n💾 PASO 7: Guardando estado compartido")
                save_processing_state(self.base_dir, metadata)
            
            # Resumen final
            print("\n" + "="*70)
            print("✅ PROCESAMIENTO COMPLETADO EXITOSAMENTE")
            print("="*70)
            print(f"\n📁 Archivos generados en: {self.base_dir}/")
            print(f"  • Video: {self.video_file.name}")
            print(f"  • Audio: {self.audio_file.name}")
            print(f"  • Intervalos: {self.intervals_file.name}")
            print(f"  • Keypoints: {self.keypoints_file.name}")
            
            if CONFIG_AVAILABLE:
                print(f"\n📋 Estado guardado para predictor")
            
            print("\n" + "="*70)
            print("🎯 SIGUIENTE PASO:")
            print("   Ejecuta el predictor para analizar las señas detectadas")
            print("   → python predictor.py --mode state")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ ERROR EN EL PROCESAMIENTO: {e}")
            import traceback
            traceback.print_exc()
            raise


# ======================================================
# FUNCIONES DE ALTO NIVEL
# ======================================================
def process_from_url(url, times_between=None, visualize=False, skip_transcription=False):
    """
    Procesa un video desde URL.
    
    Args:
        url: URL del video
        times_between: Duración de segmentos en segundos
        visualize: Mostrar visualización
        skip_transcription: Omitir transcripción
    """
    if CONFIG_AVAILABLE:
        ensure_directories()
    
    sign_segmenter = SignSegments()
    task = VideoTask(url=url)
    task.process(sign_segmenter, times_between, visualize, skip_transcription)

def process_from_file(video_path, times_between=None, visualize=False, skip_transcription=False):
    """
    Procesa un video desde archivo local.
    
    Args:
        video_path: Ruta al archivo de video
        times_between: Duración de segmentos en segundos
        visualize: Mostrar visualización
        skip_transcription: Omitir transcripción
    """
    if CONFIG_AVAILABLE:
        ensure_directories()
    
    sign_segmenter = SignSegments()
    task = VideoTask(video_path=video_path)
    task.process(sign_segmenter, times_between, visualize, skip_transcription)


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Miner - Procesador de videos LSM')
    parser.add_argument('--url', type=str, help='URL del video a descargar')
    parser.add_argument('--video', type=str, help='Ruta al video local')
    parser.add_argument('--duration', type=float, help='Duración de cada segmento en segundos')
    parser.add_argument('--visualize', action='store_true', help='Mostrar visualización')
    parser.add_argument('--skip-transcription', action='store_true', help='Omitir transcripción')
    
    args = parser.parse_args()
    
    # Cargar variables de entorno
    load_dotenv()
    
    print("\n" + "="*70)
    print("🚀 DATA MINER - PROCESADOR DE VIDEOS LSM")
    print("="*70)
    
    if CONFIG_AVAILABLE:
        ensure_directories()
        print("✓ Sistema de configuración compartida activo")
    
    try:
        # Crear segmentador
        sign_segmenter = SignSegments()
        
        # Determinar fuente del video
        if args.url:
            print(f"\n📍 Modo: Descargar desde URL")
            print(f"   URL: {args.url}")
            task = VideoTask(url=args.url)
        elif args.video:
            print(f"\n📍 Modo: Procesar archivo local")
            print(f"   Video: {args.video}")
            task = VideoTask(video_path=args.video)
        else:
            # Usar URL por defecto para demo
            print(f"\n📍 Modo: Demo con URL por defecto")
            default_url = "https://youtu.be/3rtF7_1xb9A?si=OgV1ggUnXcpGZhey"
            print(f"   URL: {default_url}")
            task = VideoTask(url=default_url)
        
        # Procesar
        task.process(
            sign_segmenter,
            times_between=args.duration,
            visualize=args.visualize,
            skip_transcription=args.skip_transcription
        )
    
    except KeyboardInterrupt:
        print("\n\n⚠ Proceso interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)