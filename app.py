import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
from datetime import datetime

model = YOLO("best.pt")

detection_log = []

def detect_image(image):
    if image is None:
        return None, "No image uploaded", ""
    
    results = model(image)[0]
    annotated = results.plot()
    count = len(results.boxes)
    
    # detection summary
    summary = f"Detected {count} garbage instance(s)"
    
    # detection log
    timestamp = datetime.now().strftime("%H:%M:%S")
    detection_log.append(f"[{timestamp}] Image — {count} garbage detected")
    log_text = "\n".join(detection_log[-10:])
    
    return annotated, summary, log_text


def detect_video(video):
    if video is None:
        return None
    
    results = model(video, save=True, project="runs", name="video_out")
    output_path = results[0].save_dir + "/video_out.mp4"
    return output_path


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
        gr.Markdown("Live webcam detection.")
        webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam Feed", type="numpy")
        webcam_output = gr.Image(label="Detection Output")
        webcam_summary = gr.Textbox(label="Live Count")
        
        webcam_input.stream(
            fn=detect_image,
            inputs=webcam_input,
            outputs=[webcam_output, webcam_summary, gr.Textbox(visible=False)]
        )

demo.launch()