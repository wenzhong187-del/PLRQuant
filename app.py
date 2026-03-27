import sys
import os
import torch
import tempfile
import logging
import traceback
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import gradio as gr
from modelscope.hub.snapshot_download import snapshot_download

sys.path.append(os.getcwd())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from model import UDTransNet
    from config import get_model_config
except Exception as e:
    logger.error(f"Error: {e}")

def get_model():
    try:
        model_dir = snapshot_download("peiyingzhong/UDTransNet")
        model_path = os.path.join(model_dir, "best_model-UDTransNet.pth.tar")
        config = get_model_config()
        model = UDTransNet(config=config, img_size=224)
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        return model, device
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        raise

model, device = get_model() if 'get_model' in locals() else (None, None)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess_for_segmentation(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.uint8(np.clip(cv2.magnitude(sobelx, sobely), 0, 255))
    enhanced = cv2.addWeighted(enhanced, 0.8, edges, 0.2, 0)
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB))

def segment_image(pil_img):
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    mask_prob = output.squeeze().cpu().numpy()
    return (mask_prob > 0.5).astype(np.uint8) * 255

def process_single_image_core(input_filepath):
    try:
        img_bgr = cv2.imread(input_filepath)
        orig_h, orig_w = img_bgr.shape[:2]
        mask = segment_image(preprocess_for_segmentation(img_bgr))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        final_mask = cv2.resize(mask_closed, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area, diameter, aspect_ratio = 0, 0, 1.0
        display_img = img_bgr.copy()
        ellipse_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            diameter = 2 * np.sqrt(area / np.pi) if area > 0 else 0
            if len(largest) >= 5:
                (x, y), (MA, ma), angle = cv2.fitEllipse(largest)
                aspect_ratio = max(MA, ma) / min(MA, ma) if min(MA, ma) > 0 else 1.0
                cv2.ellipse(display_img, ((int(x), int(y)), (int(MA), int(ma)), angle), (255, 255, 0), 2)
                cv2.ellipse(ellipse_mask, ((int(x), int(y)), (int(MA), int(ma)), angle), (255, 255, 255), -1)
        temp_dir = tempfile.mkdtemp()
        base = os.path.splitext(os.path.basename(input_filepath))[0]
        res_path = os.path.join(temp_dir, f"{base}_annotated.jpg")
        mask_path = os.path.join(temp_dir, f"{base}_ellipse_mask.jpg")
        cv2.imwrite(res_path, display_img)
        cv2.imwrite(mask_path, ellipse_mask)
        return {'Image_Name': os.path.basename(input_filepath), 'Pupil_Area_px': f"{area:.2f}", 
                'Eq_Diameter_px': f"{diameter:.2f}", 'Aspect_Ratio': f"{aspect_ratio:.3f}", 
                'Status': 'Success'}, res_path, mask_path
    except Exception as e:
        return {'Image_Name': os.path.basename(input_filepath), 'Status': f"Error: {str(e)}"}, None, None

def process_single_video(video_path, frame_interval, stimulus_start):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    analysis_data, mask_history = [], []
    frame_idx = 0
    palette = [(255,0,0), (255,255,0), (0,255,0), (144,238,144), (0,215,255), (0,165,255), (0,0,255), (226,43,138)]
    while True:
        ret, frame = cap.read()
        if not ret: break
        curr_area = 0
        if frame_idx % frame_interval == 0:
            mask = segment_image(preprocess_for_segmentation(frame))
            mask_history.append(mask)
            if len(mask_history) > 3: mask_history.pop(0)
        if mask_history:
            avg_mask = cv2.resize(cv2.GaussianBlur(np.mean(mask_history, axis=0).astype(np.uint8), (5,5), 0), (w, h))
            cnts, _ = cv2.findContours(avg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                largest = max(cnts, key=cv2.contourArea)
                curr_area = cv2.contourArea(largest)
                if len(largest) >= 5:
                    (xc, yc), (d1, d2), ang = cv2.fitEllipse(largest)
                    for i in range(8):
                        theta = i * (np.pi / 4)
                        px = int(xc + (d1/2)*np.cos(theta)*np.cos(np.radians(ang)) - (d2/2)*np.sin(theta)*np.sin(np.radians(ang)))
                        py = int(yc + (d1/2)*np.cos(theta)*np.sin(np.radians(ang)) + (d2/2)*np.sin(theta)*np.cos(np.radians(ang)))
                        cv2.circle(frame, (px, py), 12, palette[i], -1)
                M = cv2.moments(largest)
                if M["m00"] > 0: cv2.circle(frame, (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])), 5, (0,0,255), -1)
        analysis_data.append({'Time_s': frame_idx/fps, 'Pupil_Area_px': curr_area})
        out.write(frame)
        frame_idx += 1
    cap.release(); out.release()
    df = pd.DataFrame(analysis_data)
    df['Pupil_Area_px'] = df['Pupil_Area_px'].replace(0, np.nan).interpolate().bfill()
    df['Smooth_Area'] = df['Pupil_Area_px'].rolling(5, center=True).mean().fillna(df['Pupil_Area_px'])
    base_area = df[df['Time_s'] < stimulus_start]['Smooth_Area'].mean() if not df[df['Time_s'] < stimulus_start].empty else df['Smooth_Area'].iloc[0]
    post_df = df[df['Time_s'] >= stimulus_start].copy()
    metrics = {}
    if not post_df.empty:
        react_df = post_df[post_df['Smooth_Area'] < base_area * 0.97]
        latency = react_df.iloc[0]['Time_s'] - stimulus_start if not react_df.empty else 0
        min_idx = post_df['Smooth_Area'].idxmin()
        min_area = post_df.loc[min_idx, 'Smooth_Area']
        t75_target = min_area + 0.75 * (base_area - min_area)
        rec_df = post_df[(post_df['Time_s'] > post_df.loc[min_idx, 'Time_s']) & (post_df['Smooth_Area'] >= t75_target)]
        t75_time = rec_df.iloc[0]['Time_s'] - post_df.loc[min_idx, 'Time_s'] if not rec_df.empty else np.nan
        metrics = {'Latency_s': max(0, latency), 'Baseline_Area': base_area, 'Min_Constriction': min_area,
                   'Constriction_Percent': (1 - min_area/base_area)*100, 'T75_Recovery_Time': t75_time}
    df.attrs['PLR_metrics'] = metrics
    return tmp_out.name, df

def create_interface():
    css = "h1 { color: #004080; text-align: center; font-size: 48px !important; font-weight: 800 !important; } .primary_btn { background-color: #8A2BE2 !important; color: white !important; }"
    with gr.Blocks(title="PupilAI Platform", css=css) as demo:
        gr.Markdown("# PupilAI Analysis Platform\n**Automated Framework for Pupillary Light Reflex Analysis**")
        with gr.Tabs():
            with gr.TabItem("🎥 Video Batch Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        v_in = gr.File(label="Upload Videos", file_count="multiple")
                        f_step = gr.Slider(1, 30, value=5, label="Frame Interval")
                        s_time = gr.Number(value=5, label="Stimulus Start (s)")
                        btn_v = gr.Button("▶️ Run Analysis", variant="primary", elem_classes="primary_btn")
                    with gr.Column(scale=2):
                        v_out = gr.Video(label="Processed Preview")
                        v_plot = gr.Plot(label="PLR Kinetic Curve")
                    with gr.Column(scale=1):
                        v_csv = gr.File(label="Metrics CSV")
                        v_zip = gr.File(label="Annotated Results ZIP")
            with gr.TabItem("🖼️ Static Image Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        i_in = gr.File(label="Upload Images", file_count="multiple")
                        btn_i = gr.Button("▶️ Process Images", variant="primary", elem_classes="primary_btn")
                    with gr.Column(scale=2):
                        with gr.Row():
                            i_ann = gr.Image(label="Annotated Image")
                            i_mask = gr.Image(label="Binary Mask")
                        i_sel = gr.Dropdown(label="Select Image to Preview")
    return demo

if __name__ == "__main__":
    create_interface().launch()
