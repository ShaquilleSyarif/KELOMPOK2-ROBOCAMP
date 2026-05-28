import cv2
import time
from ultralytics import YOLO


MODEL_PATH = 'model_asv_2024/bola.pt' 


CLASS_ID_MERAH = 1
CLASS_ID_HIJAU = 0
CLASS_ID_OBSTACLE = 99 

CONF_THRESHOLD = 0.35
VIDEO_SOURCE = 'video/video.mp4' 


PIXELS_TO_METER = 0.02 
DEADZONE_PX = 40 


COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_YELLOW = (0, 255, 255)

# UI
def draw_hud(frame, decision, count_red, count_green, count_obs, fps, koreksi_text):
    """Menggambar panel informasi di pojok kiri atas"""
   
    cv2.rectangle(frame, (10, 10), (320, 220), (80, 80, 80), -1)
    
    # Warna teks arah
    if decision == "FORWARD": arah_color = COLOR_GREEN
    elif decision in ["LEFT", "RIGHT"]: arah_color = COLOR_CYAN
    else: arah_color = COLOR_ORANGE

    # Teks HUD 
    cv2.putText(frame, f"ARAH : {decision}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, arah_color, 2)
    cv2.putText(frame, f"Buoy Merah : {count_red}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 1)
    cv2.putText(frame, f"Buoy Hijau : {count_green}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 1)
    cv2.putText(frame, f"Obstacle     : {count_obs}", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 1)
    cv2.putText(frame, f"FPS : {fps:.1f}", (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Teks Koreksi Jarak
    if koreksi_text:
        cv2.putText(frame, koreksi_text, (15, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

def draw_bbox_with_label(frame, box, label, confidence, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, color, -1)
    
    text = f"{label} {int(confidence * 100)}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), (0, 0, 0), -1)
    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return (cx, cy)


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print("Gagal membuka video/kamera.")
        return

    window_name = "ASV Navigation System - RoboCamp 2026 (q=quit, p=pause, +/-=speed)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    prev_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        red_buoys, green_buoys = [], []
        count_obs = 0

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                coords = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()

                if cls_id == CLASS_ID_MERAH:
                    center = draw_bbox_with_label(frame, coords, "red", conf, COLOR_RED)
                    red_buoys.append(center)
                elif cls_id == CLASS_ID_HIJAU:
                    center = draw_bbox_with_label(frame, coords, "green", conf, COLOR_GREEN)
                    green_buoys.append(center)
                elif cls_id == CLASS_ID_OBSTACLE:
                    draw_bbox_with_label(frame, coords, "obstacle", conf, COLOR_ORANGE)
                    count_obs += 1

      
        decision = "FORWARD"
        koreksi_hud = ""
        banner_text = ""
        offset_real_m = 0.0
        
        
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        
        cv2.circle(frame, (frame_center_x, frame_center_y), 8, COLOR_YELLOW, -1)

        if red_buoys and green_buoys:
            nearest_red = max(red_buoys, key=lambda b: b[1])
            nearest_green = max(green_buoys, key=lambda b: b[1])

            mid_x = (nearest_red[0] + nearest_green[0]) // 2
            mid_y = (nearest_red[1] + nearest_green[1]) // 2
            midpoint = (mid_x, mid_y)

           
            cv2.line(frame, nearest_red, midpoint, COLOR_CYAN, 3)
            cv2.line(frame, nearest_green, midpoint, COLOR_CYAN, 3)
            
            
            cv2.line(frame, (frame_center_x, frame_center_y), midpoint, COLOR_CYAN, 1)

            
            cv2.circle(frame, midpoint, 10, COLOR_CYAN, -1)
            cv2.circle(frame, midpoint, 12, (255, 255, 255), 2)
            
            text_mid = "MIDPOINT"
            (tw, th), _ = cv2.getTextSize(text_mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (mid_x + 15, mid_y - th - 5), (mid_x + 15 + tw, mid_y + 5), (0,0,0), -1)
            cv2.putText(frame, text_mid, (mid_x + 15, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CYAN, 2)

            
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

        if count_obs > 0:
            decision = "AVOID/STOP"
            banner_text = "!!! BAHAYA OBSTACLE - HENTIKAN KAPAL !!!"

        
        draw_hud(frame, decision, len(red_buoys), len(green_buoys), count_obs, fps, koreksi_hud)

        
        if banner_text:
            (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            banner_x = (frame.shape[1] - tw) // 2
            banner_y = frame.shape[0] - 60
            
          
            cv2.rectangle(frame, (banner_x - 20, banner_y - th - 10), (banner_x + tw + 20, banner_y + 10), (0,0,0), -1)
            cv2.putText(frame, banner_text, (banner_x, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_YELLOW, 2)
            
            
            offset_detail = f"Offset: {offset_real_m:.2f}m"
            (tw2, _), _ = cv2.getTextSize(offset_detail, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(frame, offset_detail, ((frame.shape[1] - tw2) // 2, banner_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()