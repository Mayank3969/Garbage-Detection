import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
from datetime import datetime
import os
import cv2

model = YOLO("best.pt")

detection_log = []
frame_counter = 0
FRAME_SKIP = 3


def detect_image(image):
    if image is None:
        return None, "No image uploaded", ""

    results = model(image)[0]
    annotated = results.plot()
    count = len(results.boxes)

    summary = f"Detected {count} garbage instance(s)"

    timestamp = datetime.now().strftime("%H:%M:%S")
    detection_log.append(f"[{timestamp}] Image — {count} garbage detected")
    log_text = "\n".join(detection_log[-10:])

    return annotated, summary, log_text


def detect_video(video):
    if video is None:
        return None

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = "runs/video_out/output.mp4"
    os.makedirs("runs/video_out", exist_ok=True)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    last_annotated = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % FRAME_SKIP == 0:
            results = model(frame)[0]
            last_annotated = results.plot()

        if last_annotated is not None:
            out.write(last_annotated)
        else:
            out.write(frame)

        idx += 1

    cap.release()
    out.release()
    return out_path


def stream_webcam(image):
    global frame_counter

    if image is None:
        return image, "", ""

    frame_counter += 1
    if frame_counter % FRAME_SKIP != 0:
        return image, "", ""

    results = model(image)[0]
    annotated = results.plot()
    count = len(results.boxes)
    summary = f"Detected {count} garbage instance(s)"

    return annotated, summary, ""


def toggle_webcam(current_state):
    global frame_counter
    new_state = not current_state
    frame_counter = 0
    btn_label = "⏹ Stop Webcam" if new_state else "▶ Start Webcam"
    # show component when active, hide when stopped
    return new_state, btn_label, gr.update(visible=new_state)


with gr.Blocks(title="Garbage Detection System") as demo:
    gr.Markdown("# 🗑️ Garbage Detection System")
    gr.Markdown("Detect garbage in images and video using YOLOv8n trained on 2405 images.")

    with gr.Tab("Image Upload"):
        with gr.Row():
            img_input = gr.Image(label="Upload Image", type="numpy")
            img_output = gr.Image(label="Detection Result")

        summary_box = gr.Textbox(label="Detection Summary")
        log_box = gr.Textbox(label="Detection Log", lines=5)
        detect_btn = gr.Button("Detect Garbage", variant="primary")

        detect_btn.click(
            fn=detect_image,
            inputs=img_input,
            outputs=[img_output, summary_box, log_box]
        )

    with gr.Tab("Video Upload"):
        gr.Markdown("Upload a video — detections overlaid frame by frame.")
        vid_input = gr.Video(label="Upload Video")
        vid_output = gr.Video(label="Detection Result")
        vid_btn = gr.Button("Detect Garbage in Video", variant="primary")

        vid_btn.click(
            fn=detect_video,
            inputs=vid_input,
            outputs=vid_output
        )

    with gr.Tab("Webcam"):
        gr.Markdown("Live webcam detection. Press Start to begin, Stop to release camera.")

        is_active_state = gr.State(False)
        toggle_btn = gr.Button("▶ Start Webcam", variant="primary")

        # hidden by default — visible=False releases camera
        webcam_input = gr.Image(
            sources=["webcam"],
            streaming=True,
            label="Webcam Feed",
            type="numpy",
            visible=False
        )
        webcam_output = gr.Image(label="Detection Output")
        webcam_summary = gr.Textbox(label="Live Count")
        webcam_log_hidden = gr.Textbox(visible=False)

        toggle_btn.click(
            fn=toggle_webcam,
            inputs=is_active_state,
            outputs=[is_active_state, toggle_btn, webcam_input]
        )

        webcam_input.stream(
            fn=stream_webcam,
            inputs=webcam_input,
            outputs=[webcam_output, webcam_summary, webcam_log_hidden]
        )

demo.launch()