import base64
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st


WORKFLOW_URL = (
    "https://serverless.roboflow.com/infer/workflows/"
    "theftddetection-lwj20/"
    "waterlogging-video-bounding-boxes-1789221984773"
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


def extract_first_output(response_json):
    """Handle common Roboflow Workflow response wrappers."""
    if isinstance(response_json, list):
        if not response_json:
            raise RuntimeError("The Workflow returned an empty response.")
        return response_json[0]

    if not isinstance(response_json, dict):
        raise RuntimeError("Unexpected Workflow response format.")

    if "outputs" in response_json:
        outputs = response_json["outputs"]

        if isinstance(outputs, list):
            if not outputs:
                raise RuntimeError("The Workflow returned no outputs.")
            return outputs[0]

        if isinstance(outputs, dict):
            return outputs

    return response_json


def decode_workflow_image(value):
    """Decode the Workflow's output_image value into a BGR frame."""
    if isinstance(value, dict):
        value = (
            value.get("value")
            or value.get("image")
            or value.get("base64")
        )

    if not isinstance(value, str) or not value:
        raise RuntimeError("The Workflow returned no output_image.")

    if value.startswith("data:image"):
        value = value.split(",", 1)[1]

    value += "=" * ((4 - len(value) % 4) % 4)

    image_bytes = base64.b64decode(value)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError("Could not decode the annotated frame.")

    return frame


def prediction_count(value):
    if isinstance(value, list):
        return len(value)

    if isinstance(value, dict):
        predictions = value.get("predictions", [])
        return len(predictions) if isinstance(predictions, list) else 0

    return 0


def run_workflow_on_frame(frame, api_key):
    """Send one video frame to the deployed Roboflow Workflow."""
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:
        raise RuntimeError("Could not encode an input frame.")

    frame_base64 = base64.b64encode(encoded).decode("utf-8")

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
            f"Roboflow request failed ({response.status_code}): {message}"
        )

    output = extract_first_output(response.json())
    annotated_frame = decode_workflow_image(output.get("output_image"))
    detections = prediction_count(output.get("predictions"))

    return annotated_frame, detections


def make_browser_video(rendered_path, original_path, final_path):
    """
    Convert to H.264 and retain the original audio when available.
    The ? on the audio mapping makes audio optional.
    """
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


def process_video(input_path, output_path, api_key):
    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        fps = 25.0

    rendered_path = input_path.parent / "rendered.mp4"
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

            annotated, detections = run_workflow_on_frame(
                frame=frame,
                api_key=api_key,
            )

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
            total_detections += detections

            if total_frames > 0:
                progress.progress(
                    min(processed / total_frames, 1.0)
                )

            status.write(
                f"Processing frame {processed}"
                + (
                    f" of {total_frames}"
                    if total_frames > 0
                    else ""
                )
            )

    finally:
        capture.release()

        if writer is not None:
            writer.release()

    if processed == 0:
        raise RuntimeError("No video frames were processed.")

    status.write("Encoding the downloadable video…")

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


uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_video is not None:
    st.subheader("Input video")
    st.video(uploaded_video)

    if st.button(
        "Detect Waterlogging",
        type="primary",
        use_container_width=True,
    ):
        api_key = st.secrets.get("ROBOFLOW_API_KEY")

        if not api_key:
            st.error(
                "ROBOFLOW_API_KEY is missing. Add it under "
                "Streamlit App settings → Secrets."
            )
            st.stop()

        suffix = Path(uploaded_video.name).suffix or ".mp4"

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / f"input{suffix}"
            output_path = directory / "waterlogging_result.mp4"

            input_path.write_bytes(uploaded_video.getbuffer())

            try:
                summary = process_video(
                    input_path=input_path,
                    output_path=output_path,
                    api_key=api_key,
                )

                video_bytes = output_path.read_bytes()

                st.success("Annotated video generated successfully.")

                left, right = st.columns(2)
                left.metric("Frames processed", summary["frames"])
                right.metric(
                    "Total detections",
                    summary["detections"],
                )

                st.subheader("Annotated result")
                st.video(video_bytes)

                st.download_button(
                    "Download annotated video",
                    data=video_bytes,
                    file_name="waterlogging_detected.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

            except requests.Timeout:
                st.error(
                    "A Roboflow request timed out. Try a shorter "
                    "or lower-resolution video."
                )

            except subprocess.CalledProcessError as error:
                details = error.stderr.decode(
                    "utf-8",
                    errors="replace",
                )[-1500:]

                st.error(f"Video encoding failed:\n{details}")

            except Exception as error:
                st.error(f"Processing failed: {error}")
