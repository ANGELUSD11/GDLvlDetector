import os
import cv2

def extract_frames(video_path, output_folder, interval=30):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_frames = 0

    while cap.isOpened(): 
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            filename = os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_frame_{saved_frames}.jpg")
            cv2.imwrite(filename, frame)
            saved_frames += 1

        frame_count += 1

    cap.release()

# Procesar todos los videos en la carpeta "videos"
videos_folder = "videos"
output_folder = "vidframes"
interval = 30

for filename in os.listdir(videos_folder):
    if filename.endswith((".mp4", ".avi", ".mov")):  # formatos comunes
        video_path = os.path.join(videos_folder, filename)
        extract_frames(video_path, output_folder, interval)
print(f"Extracción de frames completada para todos los videos en {videos_folder}")
