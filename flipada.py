import os
import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import hashlib
from huggingface_hub import hf_hub_download
import requests

# Configuración de Streamlit
st.set_page_config(
    page_title="Basketball YOLO Detection", 
    page_icon=":basketball:",
    layout="wide"
)

# Token de acceso a Hugging Face (se recomienda usar variables de entorno en producción)
HF_TOKEN = os.getenv("HF_TOKEN")

# Función para descargar archivos desde Hugging Face (privados)
def descargar_archivo_hf(repo_id, filename, save_path):
    if not os.path.exists(save_path):
        st.info(f"Descargando {filename} desde Hugging Face...")
        archivo_path = hf_hub_download(repo_id=repo_id, filename=filename, use_auth_token=HF_TOKEN)
        os.rename(archivo_path, save_path)
        st.success(f"{filename} descargado correctamente!")
    return save_path

# Repositorio y archivos en Hugging Face
HF_REPO_ID = "Marxx01/yolo_basket"
MODELOS_HF = {
    "YOLOv11m": "best_yolo11m.pt",
    "YOLOv11n": "best_yolo11n.pt",
    "YOLOv5n6u": "best_YOLOv5n6u.pt",
    "YOLOv8m": "best_yolov8m.pt",
    "YOLOv8x": "best_yolov8x.pt"
}
VIDEOS_HF = {
    "Video 1": "modelo_test_video.mp4",
    "Video 2": "modelo_train_video.mp4"
}

# Crear carpeta para modelos y videos
os.makedirs("models", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# Función para hashear contraseñas
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Usuarios predefinidos
USERS = {
    'demo': hash_password('yolo2024')
}

# Función de autenticación
def authenticate(username, password):
    return username in USERS and USERS[username] == hash_password(password)

# Función para procesar video
def process_video(video_path, model, max_frames=100):
    if model == "best_YOLOv5n6u.pt":
        size = (1280, 1280)
    else: 
        size = (640, 360)
    cap = cv2.VideoCapture(video_path)
    processed_frames = []
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, size)
        results = model(frame, stream=True, conf=0.3)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = model.names[cls]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        processed_frames.append(frame)
        frame_count += 1
    
    cap.release()
    return processed_frames

def main():
    st.title("🏀 Basketball YOLO Detection")
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if not st.session_state['logged_in']:
        st.header("Login")
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        
        if st.button("Iniciar Sesión"):
            if authenticate(username, password):
                st.session_state['logged_in'] = True
                st.success("Inicio de sesión exitoso")
            else:
                st.error("Credenciales incorrectas")
        return
    
    st.sidebar.header("Configuración de Detección")
    
    modelo_seleccionado = st.sidebar.selectbox("Seleccionar Modelo", list(MODELOS_HF.keys()))
    modelo_path = descargar_archivo_hf(HF_REPO_ID, MODELOS_HF[modelo_seleccionado], os.path.join("models", MODELOS_HF[modelo_seleccionado]))
    
    if modelo_path:
        model = YOLO(modelo_path)
        st.sidebar.success(f"Modelo {modelo_seleccionado} cargado correctamente")
    else:
        return
    
    fuente_video = st.sidebar.radio("Fuente de Video", ["Video de Carpeta", "Subir Video"])
    
    if fuente_video == "Video de Carpeta":
        video_seleccionado = st.sidebar.selectbox("Seleccionar Video", list(VIDEOS_HF.keys()))
        video_path = descargar_archivo_hf(HF_REPO_ID, VIDEOS_HF[video_seleccionado], os.path.join("videos", VIDEOS_HF[video_seleccionado]))
    else:
        uploaded_file = st.file_uploader("Elige un video", type=["mp4", "avi", "mov"], help="Máximo 200MB")
        video_path = uploaded_file.name
    
    if st.button("Procesar Video"):
        if video_path is not None:
            try:
                processed_frames = process_video(video_path, model)
                st.header(f"Frames Procesados - {modelo_seleccionado}")
                for frame in processed_frames:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            except Exception as e:
                st.error(f"Error al procesar video: {e}")
        else:
            st.warning("Por favor, selecciona un video")
    
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state['logged_in'] = False
        st.experimental_rerun()

if __name__ == "__main__":
    main()
