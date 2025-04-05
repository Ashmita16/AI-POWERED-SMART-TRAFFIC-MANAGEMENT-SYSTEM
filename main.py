import streamlit as st
import cv2
import tempfile
import numpy as np
import time
import pytesseract
from ultralytics import YOLO
from sort import Sort  


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

model = YOLO("yolov8s.pt")

tracker = Sort()

EMERGENCY_CLASSES = [3, 4, 5] 

def detect_forward_moving_cars(frame, tracked_objects):
    results = model.predict(frame, conf=0.4)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else np.array([])

    car_detections = [detection[:5].tolist() for detection in detections if int(detection[5]) == 2]  
    car_detections = np.array(car_detections) if car_detections else np.empty((0, 5))

    tracked_objects_arr = tracker.update(car_detections)
    forward_count = 0
    new_tracked_objects = {}

    for obj in tracked_objects_arr:
        x1, y1, x2, y2, track_id = map(int, obj)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if track_id in tracked_objects and cy > tracked_objects[track_id][1]: 
            forward_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        new_tracked_objects[track_id] = (cx, cy)

    return frame, forward_count, new_tracked_objects

def detect_emergency_vehicles(frame, detections):
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        cls = int(cls)
        roi = frame[int(y1):int(y2), int(x1):int(x2)]


        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        text = pytesseract.image_to_string(roi_thresh).strip()
        if cls in EMERGENCY_CLASSES or "AMBULANCE" in text.upper():
            return (x1, y1, x2, y2)
    return None

def traffic_light_timing(car_count, emergency_detected=False):
    green_time = 90 if emergency_detected else min(20 + (car_count * 1.5), 90)
    yellow_time = 5
    red_time = max(30, 120 - (green_time + yellow_time))
    return green_time, yellow_time, red_time

st.title("🚦 AI-POWERED SMART TRAFFIC MANAGEMENT SYSTEM")
option = st.sidebar.radio("Select Functionality:", ("Upload Video", "Upload Image"))

if option == "Upload Video":
    st.subheader("Detection of Forward Moving Cars & Adjustment of Signals")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="video")
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        st_signal = st.empty()
        tracked_objects = {}
        frame_skip = 3

        while cap.isOpened():
            for _ in range(frame_skip):
                ret, frame = cap.read()
                if not ret:
                    break

            frame = cv2.resize(frame, (800, 450))
            detected_frame, car_count, tracked_objects = detect_forward_moving_cars(frame, tracked_objects)
            green_time, yellow_time, red_time = traffic_light_timing(car_count)
            detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            stframe.image(detected_frame, caption=f"🚗 Forward-moving cars: {car_count}", use_container_width=True)
            st_signal.markdown(f"""
            ### 🚦 Dynamic Traffic Signal Timings
            - Vehicles detected: `{car_count}`
            - 🟢 Green Light: `{green_time:.2f} sec`
            - 🟡 Yellow Light: `{yellow_time} sec`
            - 🔴 Red Light: `{red_time:.2f} sec`
            """)
            st.warning("🚨 No emergency vehicle detected in the current frame.")

            time.sleep(0.5)
        cap.release()

elif option == "Upload Image":
    st.subheader("Detection of Emergency Vehicles and Adjustment of Signals")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image")
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.resize(image, (800, 450))
        results = model.predict(image, conf=0.4)
        detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else np.array([])
        emergency_box = detect_emergency_vehicles(image, detections)
        emergency_detected = emergency_box is not None
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection.tolist()
            color = (0, 0, 255) if emergency_detected and (x1, y1, x2, y2) == emergency_box else (0, 255, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="🚗 Vehicles Detected", use_container_width=True)
        green_time, yellow_time, red_time = traffic_light_timing(0, emergency_detected)
        st.markdown(f"""
        ### 🚦 Updated Traffic Signal Timings
        - Emergency Vehicle Detected: `{emergency_detected}`
        - 🟢 Green Light: `{green_time:.2f} sec`
        - 🟡 Yellow Light: `{yellow_time} sec`
        - 🔴 Red Light: `{red_time:.2f} sec`
        """)
        st.success("Traffic signal adjusted for emergency vehicle priority.")
