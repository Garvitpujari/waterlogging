import base64
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.webrtc import StreamConfig, VideoFileSource


WORKSPACE = "theftddetection-lwj20"
WORKFLOW_ID = "waterlogging-video-bounding-boxes-1789221984773"
API_URL = "https://serverless.roboflow.com"
VIDEO_OUTPUT = "output_image"


st.set_page_config(
    page_title="Waterlogging Detection",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 Waterlogging Detection")
st.write(
    "Upload a video to detect waterlogged areas and generate an "
    "annotated video with bounding boxes."
)


def decode_workflow_image(value):
    """Decode an image returned by a Roboflow Workflow."""
    if value is None:
        return None

    if isinstance(value, dict):
        value = (
            value.get("value")
            or value.get("image")
            or value.get("base64")
        )

    if not isinstance(value, str):
        return None

    if value.startswith("data:image"):
        value = value.split(",", 1)[1]

    # Restore missing Base64 padding if necessary.
    value += "=" * ((4 - len(value) % 4) % 4)

    image_bytes = base64.b64decode(value)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def convert_for_browser(source_path, destination_path):
    """Convert the intermediate video to browser-compatible H.264."""
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(destination_path),
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def process_video(input_path, output_path, api_key):
    capture = cv2.VideoCapture(str(input_path))
    input_fps = capture.get(cv2.CAP_PROP_FPS)
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if not input_fps or input_fps <= 0:
        input_fps = 25.0

    intermediate_path = Path(tempfile.mktemp(suffix=".mp4"))
    state = {
        "writer": None,
        "frames": 0,
        "detections": 0,
    }

    client = InferenceHTTPClient(
        api_url=API_URL,
        api_key=api_key,
    ).configure(
        InferenceConfiguration(api_key_transport="header")
    )

    source = VideoFileSource(
        str(input_path),
        realtime_processing=False,
    )

    config = StreamConfig(
        stream_output=[],
        data_output=[VIDEO_OUTPUT, "predictions"],
    )

    session = client.webrtc.stream(
        source=source,
        workflow=WORKFLOW_ID,
        workspace=WORKSPACE,
        image_input="image",
        config=config,
    )

    @session.on_data()
    def receive_frame(data, metadata):
        frame = decode_workflow_image(data.get(VIDEO_OUTPUT))

        if frame is None:
            return

        if state["writer"] is None:
            height, width = frame.shape[:2]
            state["writer"] = cv2.VideoWriter(
                str(intermediate_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                input_fps,
                (width, height),
            )

        state["writer"].write(frame)
        state["frames"] += 1

        predictions = data.get("predictions", [])
        if isinstance(predictions, dict):
            predictions = predictions.get("predictions", [])

        if isinstance(predictions, list):
            state["detections"] += len(predictions)

    try:
        session.run()
    finally:
        if state["writer"] is not None:
            state["writer"].release()

    if state["frames"] == 0:
        raise RuntimeError(
            "The Workflow returned no annotated video frames."
        )

    try:
        convert_for_browser(intermediate_path, output_path)
    finally:
        intermediate_path.unlink(missing_ok=True)

    return {
        "frames_processed": state["frames"],
        "expected_frames": expected_frames,
        "detections": state["detections"],
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
                "ROBOFLOW_API_KEY is missing from Streamlit secrets."
            )
            st.stop()

        input_suffix = Path(uploaded_video.name).suffix or ".mp4"

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / f"input{input_suffix}"
            output_path = Path(directory) / "waterlogging_result.mp4"

            input_path.write_bytes(uploaded_video.getbuffer())

            try:
                with st.spinner(
                    "Processing video and drawing bounding boxes..."
                ):
                    summary = process_video(
                        input_path=input_path,
                        output_path=output_path,
                        api_key=api_key,
                    )

                output_bytes = output_path.read_bytes()

                st.success(
                    f"Finished processing "
                    f"{summary['frames_processed']} frames."
                )

                left, right = st.columns(2)
                left.metric(
                    "Frames processed",
                    summary["frames_processed"],
                )
                right.metric(
                    "Waterlogging detections",
                    summary["detections"],
                )

                st.subheader("Annotated video")
                st.video(output_bytes)

                st.download_button(
                    "Download annotated video",
                    data=output_bytes,
                    file_name="waterlogging_detected.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

            except Exception as error:
                st.error(f"Processing failed: {error}")
