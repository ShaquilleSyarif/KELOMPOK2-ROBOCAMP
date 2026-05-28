import cv2
import time
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# 1. KONFIGURASI MODEL & KELAS 
# ==============================================================================
MODEL_PATH = 'model_asv_2024/bola.pt' 

# Sesuaikan dengan model bola.pt kamu
CLASS_ID_MERAH = 1
CLASS_ID_HIJAU = 0
CLASS_ID_OBSTACLE = 99 # Biarkan 99 sampai kamu sudah train rintangan

CONF_THRESHOLD = 0.35
# Ganti dengan nama videomu, atau angka 0 kalau mau pakai webcam laptop
VIDEO_SOURCE = 'video/video.mp4' 

# ==============================================================================
# 2. KONFIGURASI KALIBRASI METER & UI
# ==============================================================================
PIXELS_TO_METER = 0.02 # Asumsi: 1 pixel layar = 2 cm di dunia nyata
DEADZONE_PX = 40       # Toleransi piksel sebelum kapal disuruh belok

# Warna BGR (Blue, Green, Red) untuk OpenCV
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_YELLOW = (0, 255, 255)

# ==============================================================================
# FUNGSI BANTUAN UI & RADAR BEV (Bird's Eye View)
# ==============================================================================
def draw_hud(frame, decision, count_red, count_green, count_obs, fps, koreksi_text):
    """Menggambar panel informasi di pojok kiri atas"""
    cv2.rectangle(frame, (10, 10), (320, 220), (80, 80, 80), -1)
    
    if decision == "FORWARD": arah_color = COLOR_GREEN
    elif decision in ["LEFT", "RIGHT"]: arah_color = COLOR_CYAN
    else: arah_color = COLOR_ORANGE

    cv2.putText(frame, f"ARAH : {decision}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, arah_color, 2)
    cv2.putText(frame, f"Buoy Merah : {count_red}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 1)
    cv2.putText(frame, f"Buoy Hijau : {count_green}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 1)
    cv2.putText(frame, f"Obstacle     : {count_obs}", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 1)
    cv2.putText(frame, f"FPS : {fps:.1f}", (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    if koreksi_text:
        cv2.putText(frame, koreksi_text, (15, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

def draw_bbox_with_label(frame, box, label, confidence, color):
    """Menggambar kotak deteksi dengan label background hitam biar jelas di air"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Titik tengah objek
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, color, -1)
    
    # Teks probabilitas
    text = f"{label} {int(confidence * 100)}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), (0, 0, 0), -1)
    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return (cx, cy)

def draw_radar_minimap(frame, red_boxes, green_boxes, obs_boxes, frame_w, frame_h):
    """Menggambar Radar 2D di pojok kanan bawah"""
    radar_size = 250
    # Bikin kanvas hitam buat radarnya
    radar = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
    
    # Bikin garis melingkar biar kerasa kayak radar asli
    pusat_radar = (radar_size // 2, radar_size - 20) # Posisi kapal kita (di bawah tengah radar)
    cv2.circle(radar, pusat_radar, 50, (50, 150, 50), 1)
    cv2.circle(radar, pusat_radar, 100, (50, 150, 50), 1)
    cv2.circle(radar, pusat_radar, 150, (50, 150, 50), 1)
    
    # Gambar ikon kapal kita (Segitiga Kuning)
    cv2.drawContours(radar, [np.array([[pusat_radar[0], pusat_radar[1]-10], 
                                       [pusat_radar[0]-7, pusat_radar[1]+5], 
                                       [pusat_radar[0]+7, pusat_radar[1]+5]])], 0, COLOR_YELLOW, -1)

    # Titik batas horizon (asumsi air mulai dari sepertiga layar dari atas)
    horizon_y = frame_h // 3 

    def map_to_radar(x_cam, y_cam):
        """Fungsi ajaib untuk mengubah posisi kamera depan jadi posisi radar atas"""
        r_x = int((x_cam / frame_w) * radar_size)
        y_cam_safe = max(y_cam, horizon_y) 
        jarak_persen = (y_cam_safe - horizon_y) / (frame_h - horizon_y)
        # Semakin dekat ke kapal, r_y makin besar nilainya
        r_y = int(pusat_radar[1] - (jarak_persen * (radar_size - 40)))
        return (r_x, r_y)

    # Plot Buoy Merah ke radar (pakai titik sentuh air, alias Y paling bawah)
    for box in red_boxes:
        rx, ry = map_to_radar((box[0] + box[2]) // 2, box[3])
        cv2.circle(radar, (rx, ry), 6, COLOR_RED, -1)

    # Plot Buoy Hijau ke radar
    for box in green_boxes:
        rx, ry = map_to_radar((box[0] + box[2]) // 2, box[3])
        cv2.circle(radar, (rx, ry), 6, COLOR_GREEN, -1)
        
    # Plot Obstacle ke radar
    for box in obs_boxes:
        rx, ry = map_to_radar((box[0] + box[2]) // 2, box[3])
        cv2.circle(radar, (rx, ry), 6, COLOR_ORANGE, -1)

    # Tempelin radar ke frame video asli di pojok kanan bawah
    margin = 20
    frame[frame_h - radar_size - margin : frame_h - margin, 
          frame_w - radar_size - margin : frame_w - margin] = radar

# ==============================================================================
# LOOP UTAMA PROGRAM
# ==============================================================================
def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print("Waduh bro, gagal membuka video atau kameranya. Cek lagi path-nya ya!")
        return

    window_name = "ASV Navigation System - RoboCamp 2026 (q=quit, p=pause)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    prev_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame_h, frame_w = frame.shape[:2]

        # Hitung FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # Deteksi Objek
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        # Variabel penampung data untuk logika & UI
        red_centers, green_centers = [], []
        red_raw_boxes, green_raw_boxes, obs_raw_boxes = [], [], []

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                coords = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()

                if cls_id == CLASS_ID_MERAH:
                    center = draw_bbox_with_label(frame, coords, "red", conf, COLOR_RED)
                    red_centers.append(center)
                    red_raw_boxes.append(coords) # Simpan box mentah buat radar
                elif cls_id == CLASS_ID_HIJAU:
                    center = draw_bbox_with_label(frame, coords, "green", conf, COLOR_GREEN)
                    green_centers.append(center)
                    green_raw_boxes.append(coords)
                elif cls_id == CLASS_ID_OBSTACLE:
                    draw_bbox_with_label(frame, coords, "obstacle", conf, COLOR_ORANGE)
                    obs_raw_boxes.append(coords)

        # =========================================================
        # LOGIKA MIDPOINT & INSTRUKSI NAVIGASI
        # =========================================================
        decision = "FORWARD"
        koreksi_hud = ""
        banner_text = ""
        offset_real_m = 0.0
        
        # Titik haluan kapal kita di layar kamera
        frame_center_x = frame_w // 2
        frame_center_y = frame_h // 2

        # Gambar titik kuning di tengah layar
        cv2.circle(frame, (frame_center_x, frame_center_y), 8, COLOR_YELLOW, -1)

        if red_centers and green_centers:
            # Ambil buoy yang paling dekat (Y paling besar)
            nearest_red = max(red_centers, key=lambda b: b[1])
            nearest_green = max(green_centers, key=lambda b: b[1])

            # Titik tengah lintasan
            mid_x = (nearest_red[0] + nearest_green[0]) // 2
            mid_y = (nearest_red[1] + nearest_green[1]) // 2
            midpoint = (mid_x, mid_y)

            # Gambar garis visual Midpoint
            cv2.line(frame, nearest_red, midpoint, COLOR_CYAN, 3)
            cv2.line(frame, nearest_green, midpoint, COLOR_CYAN, 3)
            cv2.line(frame, (frame_center_x, frame_center_y), midpoint, COLOR_CYAN, 1)

            cv2.circle(frame, midpoint, 10, COLOR_CYAN, -1)
            cv2.circle(frame, midpoint, 12, (255, 255, 255), 2)
            
            # Label Midpoint
            text_mid = "MIDPOINT"
            (tw, th), _ = cv2.getTextSize(text_mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (mid_x + 15, mid_y - th - 5), (mid_x + 15 + tw, mid_y + 5), (0,0,0), -1)
            cv2.putText(frame, text_mid, (mid_x + 15, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CYAN, 2)

            # Kalkulasi Jarak Offset
            offset_px = mid_x - frame_center_x
            offset_m = abs(offset_px) * PIXELS_TO_METER
            offset_real_m = offset_px * PIXELS_TO_METER 

            if offset_px < -DEADZONE_PX:
                decision = "LEFT"
                koreksi_hud = f"<= Koreksi KIRI  {offset_m:.2f}m"
                banner_text = f"<= ARAHKAN KAPAL KE KIRI  {offset_m:.2f} m"
            elif offset_px > DEADZONE_PX:
                decision = "RIGHT"
                koreksi_hud = f"Koreksi KANAN  {offset_m:.2f}m =>"
                banner_text = f"ARAHKAN KAPAL KE KANAN  {offset_m:.2f} m =>"
            else:
                decision = "FORWARD"
                koreksi_hud = "Koreksi : LURUS (Jalur Aman)"

        # Override kalau ada rintangan
        if len(obs_raw_boxes) > 0:
            decision = "AVOID/STOP"
            banner_text = "!!! BAHAYA OBSTACLE - HENTIKAN KAPAL !!!"

        # --- MENGGAMBAR UI AKHIR ---
        draw_hud(frame, decision, len(red_centers), len(green_centers), len(obs_raw_boxes), fps, koreksi_hud)
        draw_radar_minimap(frame, red_raw_boxes, green_raw_boxes, obs_raw_boxes, frame_w, frame_h)

        # Gambar Banner Hitam Besar di Bawah
        if banner_text:
            (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            banner_x = (frame_w - tw) // 2
            banner_y = frame_h - 60
            
            cv2.rectangle(frame, (banner_x - 20, banner_y - th - 10), (banner_x + tw + 20, banner_y + 10), (0,0,0), -1)
            cv2.putText(frame, banner_text, (banner_x, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_YELLOW, 2)
            
            offset_detail = f"Offset: {offset_real_m:.2f}m"
            (tw2, _), _ = cv2.getTextSize(offset_detail, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(frame, offset_detail, ((frame_w - tw2) // 2, banner_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Tampilkan Jendela Kamera
        cv2.imshow(window_name, frame)

        # Kontrol Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('p'): 
            cv2.waitKey(0) # Tekan 'P' buat nge-pause videonya (tekan tombol lain buat lanjut)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()