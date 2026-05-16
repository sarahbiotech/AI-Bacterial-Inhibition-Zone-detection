import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import io

# تحميل الموديل
model = YOLO("best.pt")

st.set_page_config(
    page_title="Inhibition Zone Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧫AI Inhibition Zone Analyzer")
st.markdown("""

This app is based on YOLOv8 model (Precision 87.6%) | May contain errors | Measurements not always accurate | Not for clinical use.
""")

# Sidebar لإدخال بيانات القرص
with st.sidebar:
    st.header("Disk Settings")
    disk_real_diameter = st.number_input("Antibiotic disk diameter (mm)", value=6.0)
    disk_pixel_diameter = st.number_input("Antibiotic disk diameter (pixels)", value=50.0)

# رفع صورة أو تصوير بالكاميرا
st.subheader("Upload or Capture Image")
uploaded_file = st.file_uploader("Upload agar plate image", type=["jpg","jpeg","png"])
camera_image = st.camera_input("Or take a photo")

image = None
if uploaded_file:
    image = Image.open(uploaded_file)
elif camera_image:
    image = Image.open(camera_image)

if image:
    image = image.resize((800,800))
    st.image(image, caption="Selected Image", use_container_width=True)

# تصنيف القطر
def classify_zone(d_mm):
    if d_mm >= 20:
        return "Sensitive"
    elif d_mm >= 15:
        return "Intermediate"
    else:
        return "Resistant"

# قياس الدائرة داخل ROI
def measure_zone_circle(roi, min_area_ratio=0.05):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < min_area_ratio * (roi.shape[0]*roi.shape[1]):
        return None

    (x, y), radius = cv2.minEnclosingCircle(largest)
    return int(radius*2), (int(x), int(y), int(radius))

# تحليل الصورة
def analyze_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img)[0]
    zones = []

    if len(results.boxes) == 0:
        return img, []

    mm_per_pixel = disk_real_diameter / disk_pixel_diameter

    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        pad = 3
        roi = img[max(y1-pad,0):y2+pad, max(x1-pad,0):x2+pad]

        result = measure_zone_circle(roi)
        if result is None:
            continue

        diameter_px, (cx, cy, r) = result
        diameter_px = min(diameter_px, max(x2-x1, y2-y1))
        diameter_mm = diameter_px * mm_per_pixel
        classification = classify_zone(diameter_mm)

        cv2.circle(img, (x1 + cx, y1 + cy), r, (255, 0, 0), 3)

        zones.append({
            "Zone": f"Zone {len(zones)+1}",
            "Diameter (px)": diameter_px,
            "Diameter (mm)": round(diameter_mm,2),
            "Result": classification
        })

    return img, zones

# زر التحليل
report_df = None
if image and st.button(" Analyze"):
    result_img, zones = analyze_image(image)
    if not zones:
        st.error("No inhibition zones detected.")
    else:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(result_img, caption="Detected Inhibition Zones", use_container_width=True)

        report_df = pd.DataFrame(zones)
        st.table(report_df)

# زر تنزيل التقرير
if report_df is not None:
    buffer = io.BytesIO()
    report_df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label=" Download Report (CSV)",
        data=buffer,
        file_name="inhibition_zones_report.csv",
        mime="text/csv"
    )

   
st.markdown(
    "<p style='text-align: center; color: gray;'>This model was Developed by Sarah</p>", 
    unsafe_allow_html=True
)




