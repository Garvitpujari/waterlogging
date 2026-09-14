import base64
import copy
import os
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="Waterlogging Detection", page_icon="🌊", layout="wide")

st.title("🌊 Waterlogging Detection")
st.caption("AI-powered waterlogging analysis using a Roboflow workflow")

WORKFLOW_URL = (
    "https://serverless.roboflow.com/infer/workflows/"
    "theftddetection-lwj20/"
    "waterlogging-video-bounding-boxes-1789221984773"
)
BACKEND_URL = "https://urban-net-sih26124.onrender.com/api/edge/events"

WATERLOGGING_EVENTS = [
    {
        "eventType": "WATERLOGGING",
        "busId": "BUS_TEST_NARELA01",
        "cameraId": "CAM_FRONT",
        "timestamp": "2026-09-14T00:10:00Z",
        "location": {"latitude": 28.8527, "longitude": 77.0926, "address": "Narela Main Road, Delhi"},
        "detection": {"confidence": 0.93, "severity": "HIGH"},
        "model": {"name": "waterlogging-yolo", "version": "1.0"},
        "evidence": {"imageUrl": None},
        "metadata": {"detectedClass": "waterlogging", "source": "streamlit-demo"},
    },
    {
        "eventType": "WATERLOGGING",
        "busId": "BUS_TEST_ROHINI02",
        "cameraId": "CAM_FRONT",
        "timestamp": "2026-09-14T00:18:32Z",
        "location": {"latitude": 28.7495, "longitude": 77.0671, "address": "Rohini Sector 7, Delhi"},
        "detection": {"confidence": 0.89, "severity": "MEDIUM"},
        "model": {"name": "waterlogging-yolo", "version": "1.0"},
        "evidence": {"imageUrl": None},
        "metadata": {"detectedClass": "waterlogging", "source": "streamlit-demo"},
    },
    {
        "eventType": "WATERLOGGING",
        "busId": "BUS_TEST_DWARKA03",
        "cameraId": "CAM_LEFT",
        "timestamp": "2026-09-14T00:27:15Z",
        "location": {"latitude": 28.5921, "longitude": 77.0460, "address": "Dwarka Sector 10, Delhi"},
        "detection": {"confidence": 0.96, "severity": "HIGH"},
        "model": {"name": "waterlogging-yolo", "version": "1.0"},
        "evidence": {"imageUrl": None},
        "metadata": {"detectedClass": "waterlogging", "source": "streamlit-demo"},
    },
    {
        "eventType": "WATERLOGGING",
        "busId": "BUS_TEST_MAYUR04",
        "cameraId": "CAM_FRONT",
        "timestamp": "2026-09-14T00:35:48Z",
        "location": {"latitude": 28.6108, "longitude": 77.2946, "address": "Mayur Vihar Phase 1, Delhi"},
        "detection": {"confidence": 0.91, "severity": "HIGH"},
        "model": {"name": "waterlogging-yolo", "version": "1.0"},
        "evidence": {"imageUrl": None},
        "metadata": {"detectedClass": "waterlogging", "source": "streamlit-demo"},
    },
]


def send_event_to_backend(event):
    """Send the event object directly to the backend."""
    try:
        response = requests.post(
            BACKEND_URL,
            headers={"Content-Type": "application/json"},
            json=event,
            timeout=30,
        )
        response.raise_for_status()
        return True, response
    except requests.RequestException as e:
        return False, e


def extract_first_output(response_json):
    if not isinstance(response_json, dict):
        return None
    outputs = response_json.get("outputs")
    if isinstance(outputs, list) and outputs:
        return outputs[0] if isinstance(outputs[0], dict) else None
    if isinstance(outputs, dict):
        return outputs
    return response_json


def decode_workflow_image(value):
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("value", "image", "data", "base64"):
            if key in value:
                return decode_workflow_image(value[key])
        return None
    if not isinstance(value, str):
        return None
    try:
        encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        image_bytes = base64.b64decode(encoded)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def prediction_count(value):
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("predictions", "detections"):
            if isinstance(value.get(key), list):
                return len(value[key])
        for key in ("output", "result", "data"):
            if key in value:
                count = prediction_count(value[key])
                if count:
                    return count
    return 0


def run_workflow_on_frame(frame, api_key):
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Could not encode video frame.")

    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    payload = {"inputs": {"image": {"type": "base64", "value": image_b64}}}

    response = requests.post(
        WORKFLOW_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    output = extract_first_output(response.json()) or {}
    output_image = None
    for key in ("output_image", "annotated_image", "visualization", "image"):
        if key in output:
            output_image = decode_workflow_image(output[key])
            if output_image is not None:
                break

    predictions = output.get("predictions") or output.get("detections") or []
    return output_image, predictions


def make_browser_video(input_path, rendered_path):
    browser_path = str(Path(rendered_path).with_name(Path(rendered_path).stem + "_browser.mp4"))
    command = [
        "ffmpeg", "-y", "-i", rendered_path, "-i", input_path,
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", browser_path,
    ]
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return browser_path
    except Exception:
        return rendered_path


def process_video(input_path, output_path, api_key):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Could not create the output video.")

    progress = st.progress(0)
    status = st.empty()
    frame_number = 0
    detection_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output_image, predictions = run_workflow_on_frame(frame, api_key)
            detection_count += prediction_count(predictions)

            if output_image is not None:
                if output_image.shape[1] != width or output_image.shape[0] != height:
                    output_image = cv2.resize(output_image, (width, height))
                annotated_frame = output_image
            else:
                annotated_frame = frame

            out.write(annotated_frame)
            frame_number += 1

            if total_frames > 0:
                progress.progress(min(frame_number / total_frames, 1.0))
                status.text(
                    f"Processing frame {frame_number} / {total_frames} • "
                    f"Waterlogging detections: {detection_count}"
                )
            else:
                status.text(
                    f"Processing frame {frame_number} • "
                    f"Waterlogging detections: {detection_count}"
                )
    finally:
        cap.release()
        out.release()

    progress.empty()
    status.empty()
    return frame_number, detection_count


# ============================================================
# MAIN APP
# ============================================================

uploaded_video = st.file_uploader(
    "🎥 Upload a waterlogging video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_video is not None:
    st.subheader("🎬 Original Video")
    st.video(uploaded_video)

    if st.button("🔍 Detect Waterlogging", type="primary", use_container_width=True):
        api_key = st.secrets.get("ROBOFLOW_API_KEY")
        if not api_key:
            st.error("❌ ROBOFLOW_API_KEY is missing from Streamlit Secrets.")
            st.stop()

        input_path = None
        output_path = None
        browser_video_path = None

        try:
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded_video.getbuffer())
                input_path = f.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                output_path = f.name

            with st.spinner("🌊 AI is analyzing the video frame-by-frame..."):
                frames_processed, total_detections = process_video(
                    input_path, output_path, api_key
                )

            st.success(f"✅ Analysis complete — {frames_processed:,} frames processed.")

            st.subheader("📊 Detection Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Frames Processed", f"{frames_processed:,}")
            with col2:
                st.metric("Total Waterlogging Detections", f"{total_detections:,}")

            incident = None

            # CRITICAL FIX: do NOT send any backend event when there
            # are zero AI detections. This is different from the old
            # pothole app shown in the screenshot, which sent an event
            # even when Total Detections was 0.
            if total_detections > 0:
                incident = copy.deepcopy(random.choice(WATERLOGGING_EVENTS))
                incident["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

                st.info("🌊 Waterlogging detected. Sending event to backend...")
                backend_success, backend_result = send_event_to_backend(incident)

                if backend_success:
                    st.success("✅ Detection event sent to backend successfully.")
                    st.caption(f"Backend HTTP status: {backend_result.status_code}")
                    try:
                        with st.expander("View Backend Response"):
                            st.json(backend_result.json())
                    except ValueError:
                        with st.expander("View Backend Response"):
                            st.code(backend_result.text or "(empty response)")
                else:
                    st.warning("⚠️ Waterlogging was detected, but the event could not be sent to backend.")
                    st.caption(f"Backend error: {backend_result}")
            else:
                st.info("ℹ️ No waterlogging detections were found. No event was sent to the backend.")

            if incident is not None:
                st.subheader("📡 Detection Event")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Event Type", incident["eventType"])
                with col2:
                    st.metric("Confidence", f'{incident["detection"]["confidence"] * 100:.0f}%')
                with col3:
                    st.metric("Severity", incident["detection"]["severity"])

                st.write("### 📍 Location")
                location = incident["location"]
                st.write(f'**Address:** {location["address"]}')
                st.write(f'**Coordinates:** {location["latitude"]}, {location["longitude"]}')

                st.write("### 🚌 Vehicle / Camera")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f'**Bus ID:** {incident["busId"]}')
                with col2:
                    st.write(f'**Camera:** {incident["cameraId"]}')

                st.write("### 🤖 Model Information")
                st.write(
                    f'**Model:** {incident["model"]["name"]}  \n'
                    f'**Version:** {incident["model"]["version"]}'
                )

                with st.expander("View Event JSON"):
                    st.json(incident)

            st.subheader("🎥 Annotated Output Video")
            browser_video_path = make_browser_video(input_path, output_path)
            if os.path.exists(browser_video_path):
                st.video(browser_video_path)
                with open(browser_video_path, "rb") as f:
                    video_bytes = f.read()
                st.download_button(
                    "⬇️ Download Annotated Video",
                    data=video_bytes,
                    file_name="waterlogging_detected.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"❌ Error while processing video: {e}")

        finally:
            for path in (input_path, output_path, browser_video_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
