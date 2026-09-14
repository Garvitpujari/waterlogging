import base64
import random
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

WORKFLOW_URL = (
    "https://serverless.roboflow.com/infer/workflows/"
    "theftddetection-lwj20/"
    "waterlogging-video-bounding-boxes-1789221984773"
)

BACKEND_URL = (
    "https://urban-net-sih26124.onrender.com/api/edge/events"
)


st.set_page_config(
    page_title="Waterlogging Detection",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 Waterlogging Detection")
st.write(
    "Upload a video to detect waterlogged areas and download "
    "the processed video with bounding boxes."
)


# ============================================================
# SAMPLE WATERLOGGING EVENTS
# ============================================================

WATERLOGGING_EVENTS = [

    {
        "eventType": "WATERLOGGING",
        "busId": "BUS_TEST_NARELA01",
        "cameraId": "CAM_FRONT",
        "timestamp": "2026-09-14T00:10:00Z",
        "location": {
            "latitude": 28.8527,
            "longitude": 77.0926,
            "address": "Narela Main Road, Delhi"
        },
        "detection": {
            "confidence": 0.93,
            "severity": "HIGH"
        },
        "model": {
            "name": "waterlogging-yolo",
            "version": "1.0"
        },
        "evidence": {
            "imageUrl": None
        },
        "metadata": {
            "detectedClass": "waterlogging",
            "source": "streamlit-demo"
        }
    },

    {
        "eventType": "WATERLOGGING",
        "busId": "BUS_TEST_ROHINI02",
        "cameraId": "CAM_FRONT",
        "timestamp": "2026-09-14T00:18:32Z",
        "location": {
            "latitude": 28.7495,
            "longitude": 77.0671,
            "address": "Rohini Sector 7, Delhi"
        },
        "detection": {
            "confidence": 0.89,
            "severity": "MEDIUM"
        },
        "model": {
            "name": "waterlogging-yolo",
            "version": "1.0"
        },
        "evidence": {
            "imageUrl": None
        },
        "metadata": {
            "detectedClass": "waterlogging",
            "source": "streamlit-demo"
        }
    },

    {
        "eventType": "WATERLOGGING",
        "busId": "BUS_TEST_DWARKA03",
        "cameraId": "CAM_LEFT",
        "timestamp": "2026-09-14T00:27:15Z",
        "location": {
            "latitude": 28.5921,
            "longitude": 77.0460,
            "address": "Dwarka Sector 10, Delhi"
        },
        "detection": {
            "confidence": 0.96,
            "severity": "HIGH"
        },
        "model": {
            "name": "waterlogging-yolo",
            "version": "1.0"
        },
        "evidence": {
            "imageUrl": None
        },
        "metadata": {
            "detectedClass": "waterlogging",
            "source": "streamlit-demo"
        }
    },

    {
        "eventType": "WATERLOGGING",
        "busId": "BUS_TEST_MAYUR04",
        "cameraId": "CAM_FRONT",
        "timestamp": "2026-09-14T00:35:48Z",
        "location": {
            "latitude": 28.6108,
            "longitude": 77.2946,
            "address": "Mayur Vihar Phase 1, Delhi"
        },
        "detection": {
            "confidence": 0.91,
            "severity": "HIGH"
        },
        "model": {
            "name": "waterlogging-yolo",
            "version": "1.0"
        },
        "evidence": {
            "imageUrl": None
        },
        "metadata": {
            "detectedClass": "waterlogging",
            "source": "streamlit-demo"
        }
    },
]


# ============================================================
# ROBOFLOW RESPONSE HELPERS
# ============================================================

def extract_first_output(response_json):
    """Handle common Roboflow Workflow response wrappers."""

    if isinstance(response_json, list):

        if not response_json:
            raise RuntimeError(
                "The Workflow returned an empty response."
            )

        return response_json[0]

    if not isinstance(response_json, dict):

        raise RuntimeError(
            "Unexpected Workflow response format."
        )

    if "outputs" in response_json:

        outputs = response_json["outputs"]

        if isinstance(outputs, list):

            if not outputs:
                raise RuntimeError(
                    "The Workflow returned no outputs."
                )

            return outputs[0]

        if isinstance(outputs, dict):
            return outputs

    return response_json


def decode_workflow_image(value):
    """Decode Workflow output_image into a BGR frame."""

    if isinstance(value, dict):

        value = (
            value.get("value")
            or value.get("image")
            or value.get("base64")
        )

    if not isinstance(value, str) or not value:

        raise RuntimeError(
            "The Workflow returned no output_image."
        )

    if value.startswith("data:image"):

        value = value.split(",", 1)[1]

    value += "=" * ((4 - len(value) % 4) % 4)

    image_bytes = base64.b64decode(value)

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        raise RuntimeError(
            "Could not decode the annotated frame."
        )

    return frame


def get_predictions(value):

    if isinstance(value, list):
        return value

    if isinstance(value, dict):

        predictions = value.get(
            "predictions",
            []
        )

        if isinstance(predictions, list):
            return predictions

    return []


# ============================================================
# RUN ROBOFLOW WORKFLOW
# ============================================================

def run_workflow_on_frame(frame, api_key):

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:

        raise RuntimeError(
            "Could not encode an input frame."
        )

    frame_base64 = base64.b64encode(
        encoded
    ).decode("utf-8")

    response = requests.post(
        WORKFLOW_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "inputs": {
                "image": {
                    "type": "base64",
                    "value": frame_base64,
                }
            }
        },
        timeout=180,
    )

    if not response.ok:

        message = response.text[:1000]

        raise RuntimeError(
            f"Roboflow request failed "
            f"({response.status_code}): {message}"
        )

    output = extract_first_output(
        response.json()
    )

    annotated_frame = decode_workflow_image(
        output.get("output_image")
    )

    predictions = get_predictions(
        output.get("predictions")
    )

    return annotated_frame, predictions


# ============================================================
# BACKEND
# ============================================================

def send_event_to_backend(event):

    """
    Send the event directly to the backend.

    Backend expects:
        POST /api/edge/events

    Body:
        event object directly

    NOT:
        {"event": event}
    """

    response = requests.post(
        BACKEND_URL,
        headers={
            "Content-Type": "application/json"
        },
        json=event,
        timeout=30,
    )

    if not response.ok:

        raise RuntimeError(
            f"Backend request failed "
            f"({response.status_code}): "
            f"{response.text[:1000]}"
        )

    return response


# ============================================================
# VIDEO ENCODING
# ============================================================

def make_browser_video(
    rendered_path,
    original_path,
    final_path,
):

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(rendered_path),
        "-i",
        str(original_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(final_path),
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    input_path,
    output_path,
    api_key,
):

    capture = cv2.VideoCapture(
        str(input_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            "Could not open the uploaded video."
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if not fps or fps <= 0:
        fps = 25.0

    rendered_path = (
        input_path.parent / "rendered.mp4"
    )

    writer = None

    processed = 0
    total_detections = 0

    progress = st.progress(0)
    status = st.empty()

    try:

        while True:

            success, frame = capture.read()

            if not success:
                break

            annotated, predictions = (
                run_workflow_on_frame(
                    frame=frame,
                    api_key=api_key,
                )
            )

            detections = len(predictions)

            total_detections += detections

            if writer is None:

                height, width = annotated.shape[:2]

                writer = cv2.VideoWriter(
                    str(rendered_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

                if not writer.isOpened():

                    raise RuntimeError(
                        "Could not create the output video."
                    )

            writer.write(annotated)

            processed += 1

            if total_frames > 0:

                progress.progress(
                    min(
                        processed / total_frames,
                        1.0
                    )
                )

            status.write(
                f"Processing frame {processed}"
                + (
                    f" of {total_frames}"
                    if total_frames > 0
                    else ""
                )
                + f" • Detections: "
                f"{total_detections}"
            )

    finally:

        capture.release()

        if writer is not None:
            writer.release()

    if processed == 0:

        raise RuntimeError(
            "No video frames were processed."
        )

    status.write(
        "Encoding the downloadable video…"
    )

    make_browser_video(
        rendered_path=rendered_path,
        original_path=input_path,
        final_path=output_path,
    )

    progress.progress(1.0)
    status.empty()

    return {
        "frames": processed,
        "detections": total_detections,
    }


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "Upload a video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv"
    ],
)


# ============================================================
# MAIN APP
# ============================================================

if uploaded_video is not None:

    st.subheader("Input video")

    st.video(uploaded_video)

    if st.button(
        "Detect Waterlogging",
        type="primary",
        use_container_width=True,
    ):

        # ----------------------------------------------------
        # GET ROBOFLOW API KEY
        # ----------------------------------------------------

        api_key = st.secrets.get(
            "ROBOFLOW_API_KEY"
        )

        if not api_key:

            st.error(
                "ROBOFLOW_API_KEY is missing. "
                "Add it under Streamlit App settings → Secrets."
            )

            st.stop()


        suffix = (
            Path(
                uploaded_video.name
            ).suffix
            or ".mp4"
        )


        with tempfile.TemporaryDirectory() as directory:

            directory = Path(directory)

            input_path = (
                directory / f"input{suffix}"
            )

            output_path = (
                directory / "waterlogging_result.mp4"
            )


            input_path.write_bytes(
                uploaded_video.getbuffer()
            )


            try:

                # =================================================
                # RUN WATERLOGGING DETECTION
                # =================================================

                summary = process_video(
                    input_path=input_path,
                    output_path=output_path,
                    api_key=api_key,
                )


                video_bytes = (
                    output_path.read_bytes()
                )


                # =================================================
                # SUMMARY
                # =================================================

                if summary["detections"] > 0:

                    st.success(
                        "🌊 Waterlogging detected."
                    )

                else:

                    st.info(
                        "No waterlogging detected "
                        "in the processed video."
                    )


                left, right = st.columns(2)

                left.metric(
                    "Frames processed",
                    summary["frames"],
                )

                right.metric(
                    "Total detections",
                    summary["detections"],
                )


                # =================================================
                # RANDOM SAMPLE EVENT
                # =================================================
                #
                # For the prototype, choose one of the predefined
                # waterlogging events randomly.
                #
                # The selected object is exactly what is sent
                # to the backend.
                # =================================================

                incident = random.choice(
                    WATERLOGGING_EVENTS
                )


                # Use current timestamp for the demo event
                incident["timestamp"] = (
                    datetime.now(
                        timezone.utc
                    )
                    .isoformat()
                    .replace("+00:00", "Z")
                )


                # =================================================
                # DISPLAY EVENT
                # =================================================

                st.subheader(
                    "📡 Detection Event"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Event Type",
                        incident["eventType"]
                    )

                with col2:

                    st.metric(
                        "Confidence",
                        f'{incident["detection"]["confidence"] * 100:.0f}%'
                    )

                with col3:

                    st.metric(
                        "Severity",
                        incident["detection"]["severity"]
                    )


                # =================================================
                # LOCATION
                # =================================================

                st.write(
                    "### 📍 Location"
                )

                location = incident["location"]

                st.write(
                    f'**Address:** '
                    f'{location["address"]}'
                )

                st.write(
                    f'**Coordinates:** '
                    f'{location["latitude"]}, '
                    f'{location["longitude"]}'
                )


                # =================================================
                # BUS / CAMERA
                # =================================================

                st.write(
                    "### 🚌 Vehicle / Camera"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        f'**Bus ID:** '
                        f'{incident["busId"]}'
                    )

                with col2:

                    st.write(
                        f'**Camera:** '
                        f'{incident["cameraId"]}'
                    )


                # =================================================
                # MODEL INFO
                # =================================================

                st.write(
                    "### 🤖 Model Information"
                )

                st.write(
                    f'**Model:** '
                    f'{incident["model"]["name"]}  \n'
                    f'**Version:** '
                    f'{incident["model"]["version"]}'
                )


                # =================================================
                # SEND TO BACKEND
                # =================================================

                try:

                    backend_response = (
                        send_event_to_backend(
                            incident
                        )
                    )

                    st.success(
                        "✅ Detection event sent "
                        "to backend successfully."
                    )

                    try:

                        backend_json = (
                            backend_response.json()
                        )

                        with st.expander(
                            "View Backend Response"
                        ):

                            st.json(
                                backend_json
                            )

                    except ValueError:

                        pass


                except requests.Timeout:

                    st.warning(
                        "⚠️ Detection completed, "
                        "but the backend request timed out."
                    )


                except Exception as error:

                    st.warning(
                        "⚠️ Detection completed, "
                        "but the event could not be sent "
                        "to the backend."
                    )

                    st.caption(
                        f"Backend error: {error}"
                    )


                # =================================================
                # EVENT JSON
                # =================================================

                with st.expander(
                    "View Event JSON"
                ):

                    st.json(
                        incident
                    )


                # =================================================
                # ANNOTATED VIDEO
                # =================================================

                st.subheader(
                    "Annotated result"
                )

                st.video(
                    video_bytes
                )


                # =================================================
                # DOWNLOAD
                # =================================================

                st.download_button(
                    "Download annotated video",
                    data=video_bytes,
                    file_name=(
                        "waterlogging_detected.mp4"
                    ),
                    mime="video/mp4",
                    use_container_width=True,
                )


            except requests.Timeout:

                st.error(
                    "A Roboflow request timed out. "
                    "Try a shorter or lower-resolution video."
                )


            except subprocess.CalledProcessError as error:

                details = error.stderr.decode(
                    "utf-8",
                    errors="replace",
                )[-1500:]

                st.error(
                    f"Video encoding failed:\n{details}"
                )


            except Exception as error:

                st.error(
                    f"Processing failed: {error}"
                )
