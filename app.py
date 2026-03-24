import sys
import gradio as gr
import os
import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import tempfile
import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from modelscope.hub.snapshot_download import snapshot_download
import traceback
import zipfile 

# ===== Key configuration and initialization (No change) =====
sys.path.append(os.getcwd())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Model import and config (No change) =====
try:
    from model import UDTransNet
    from config import get_model_config
    logger.info("✅ Successfully imported model and config files")
except Exception as e:
    logger.exception(f"❌ Failed to import model or config: {str(e)}")

# ===== Model loading function (No change) =====
def get_model():
    try:
        logger.info("===== [Model Loading Debug] Starting model download =====")
        # Download model snapshot using modelscope
        model_dir = snapshot_download("peiyingzhong/UDTransNet")
        model_path = os.path.join(model_dir, "best_model-UDTransNet.pth.tar")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        config = get_model_config()
        model = UDTransNet(config=config, img_size=224)
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Remove 'module.' prefix (for models trained with DataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        logger.info(f"✅ Model loaded successfully, Device: {device}")
        return model, device
    except Exception as e:
        logger.error(f"[❌Model Loading Failed] {str(e)}")
        logger.error(traceback.format_exc())
        print("❌ Model Loading Failed:", str(e), flush=True)
        traceback.print_exc()
        raise

# ===== Model initialization (No change) =====
try:
    model, device = get_model()
except Exception as e:
    logger.exception(f"Application initialization failed: {str(e)}")
    model, device = None, None

# ===== Image preprocessing/segmentation function (No change) =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess_for_segmentation(frame_bgr):
    # No logic change
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)
    edges = np.uint8(np.clip(edges, 0, 255))
    enhanced = cv2.addWeighted(enhanced, 0.8, edges, 0.2, 0)
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    rgb_img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_img)

def segment_image(pil_img):
    # No logic change
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    mask_prob = output.squeeze().cpu().numpy()
    mask_bin = (mask_prob > 0.5).astype(np.uint8) * 255
    return mask_bin

# **======== Single image processing core function (No logic change) ========**
def process_single_image_core(input_filepath):
    # ... (保持原函数逻辑不变) ...
    if input_filepath is None:
        return None, None, None

    try:
        # 1. Read and preprocess
        img_bgr = cv2.imread(input_filepath)
        if img_bgr is None:
            raise RuntimeError("Could not read image file.")
            
        orig_h, orig_w = img_bgr.shape[:2]
        
        preprocessed_pil_img = preprocess_for_segmentation(img_bgr)
        mask = segment_image(preprocessed_pil_img)
        
        # 2. Post-processing (smoothing, resizing)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        final_mask = cv2.resize(mask_closed, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) 
        
        # 3. Contour extraction and metrics
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area = 0
        diameter = 0
        aspect_ratio = 0
        
        display_img = img_bgr.copy()
        ellipse_mask_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8) # 3-channel mask

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if len(largest_contour) >= 5: # Needs at least 5 points for ellipse fitting
                (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)
                
                diameter = 2 * np.sqrt(area / np.pi) if area > 0 else 0
                aspect_ratio = max(MA, ma) / min(MA, ma) if min(MA, ma) > 0 else 1.0

                # **Draw fitted ellipse (Cyan) (255, 255, 0) in BGR**
                ellipse_color = (255, 255, 0) 
                cv2.ellipse(display_img, ((int(x), int(y)), (int(MA), int(ma)), angle), ellipse_color, 2)
                
                # **Draw fitted ellipse Mask (White fill)**
                cv2.ellipse(ellipse_mask_img, ((int(x), int(y)), (int(MA), int(ma)), angle), (255, 255, 255), -1) 

            else: 
                 diameter = 2 * np.sqrt(area / np.pi) if area > 0 else 0

        # 4. Format results (dictionary)
        results_dict = {
            'Image_Name': os.path.basename(input_filepath),
            'Pupil_Area_pixels': f"{area:.2f}", 
            'Equivalent_Diameter_pixels': f"{diameter:.2f}",
            'Aspect_Ratio': f"{aspect_ratio:.3f}",
            'Status': 'Success'
        }
        
        # 5. Save temporary result files
        temp_dir = tempfile.mkdtemp()
        
        base_name = os.path.splitext(os.path.basename(input_filepath))[0]
        
        # Save annotated image
        result_bgr_path = os.path.join(temp_dir, f"{base_name}_annotated.jpg")
        cv2.imwrite(result_bgr_path, display_img)
        
        # Save Mask image
        mask_path = os.path.join(temp_dir, f"{base_name}_ellipse_mask.jpg")
        cv2.imwrite(mask_path, ellipse_mask_img)

        return results_dict, result_bgr_path, mask_path
    
    except Exception as e:
        logger.error(f"Failed to process image {os.path.basename(input_filepath)}: {str(e)}")
        return {
            'Image_Name': os.path.basename(input_filepath),
            'Pupil_Area_pixels': 0, 
            'Equivalent_Diameter_pixels': 0,
            'Aspect_Ratio': 0,
            'Status': f"Failed: {str(e)}"
        }, None, None

# **======== Batch image processing function (No logic change) ========**
def process_multiple_images(image_files):
    # ... (保持原函数逻辑不变) ...
    empty_df = pd.DataFrame(columns=['Image_Name', 'Pupil_Area_pixels', 'Equivalent_Diameter_pixels', 'Aspect_Ratio', 'Status'])
    
    if not model or not device:
        dropdown_update = gr.update(choices=[], value=None, interactive=False)
        return None, None, None, dropdown_update, None, None, "Model failed to initialize. Check logs.", empty_df
    if not image_files:
        dropdown_update = gr.update(choices=[], value=None, interactive=False)
        return None, None, None, dropdown_update, None, None, "Please upload image files.", empty_df

    all_metrics = []
    annotated_paths = []
    mask_paths = []
    image_names = []
    
    progress = gr.Progress(track_tqdm=True)

    for i, file_obj in enumerate(progress.tqdm(image_files, desc="Processing Images")):
        filepath = file_obj.name
        metrics, annotated_path, mask_path = process_single_image_core(filepath)
        
        if metrics and metrics['Status'] == 'Success':
            all_metrics.append(metrics)
            image_names.append(metrics['Image_Name']) 
            annotated_paths.append(annotated_path)
            mask_paths.append(mask_path)
        else:
            if metrics:
                all_metrics.append(metrics)

    
    if not all_metrics:
        dropdown_update = gr.update(choices=[], value=None, interactive=False)
        return None, None, None, dropdown_update, None, None, "❌ All image processing failed. Check file format.", empty_df

    # 1. Generate combined metrics table CSV
    final_df = pd.DataFrame(all_metrics)
    csv_temp_file = tempfile.NamedTemporaryFile(suffix="_image_metrics.csv", delete=False)
    csv_path = csv_temp_file.name
    csv_temp_file.close()
    final_df.to_csv(csv_path, index=False)
    
    # 2. Zip all output images
    zip_temp_file = tempfile.NamedTemporaryFile(suffix="_image_results.zip", delete=False)
    zip_path = zip_temp_file.name
    zip_temp_file.close()

    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for path in annotated_paths:
                zipf.write(path, os.path.basename(path))
            for path in mask_paths:
                zipf.write(path, os.path.basename(path))
    except Exception as e:
        logger.error(f"Failed to create ZIP file: {str(e)}")
        zip_path = None
    
    # 3. Prepare Gradio Dropdown update object
    if image_names:
        dropdown_update = gr.update(
            choices=image_names, 
            value=image_names[0] if image_names else None,
            interactive=True
        )
        message = f"✅ Analysis complete for {len(all_metrics)} images. {len(image_names)} previews generated. Metrics merged and results zipped. Select an image for preview."
    else:
        dropdown_update = gr.update(choices=[], value=None, interactive=False)
        message = "❌ All image processing failed or failed to generate output files."

    # Returns: Image names list (State), annotated path list (State), Mask path list (State), Dropdown update object, CSV file, ZIP file, status message, merged Dataframe
    return image_names, annotated_paths, mask_paths, dropdown_update, csv_path, zip_path, message, final_df

# **======== Helper function: Update Dropdown Choices (No change) ========**
def update_dropdown_choices_and_value(image_names):
    # ... (保持原函数逻辑不变) ...
    if image_names:
        return gr.update(
            choices=image_names, 
            value=image_names[0], 
            interactive=True
        )
    return gr.update(choices=[], value=None, interactive=False)

# =========================================================================
# 补回被删除的预览更新函数
# =========================================================================
def update_image_preview(selected_image_name, image_names, annotated_paths, mask_paths):
    """
    用于在 Static Image 标签页中，切换下拉菜单时更新显示的图片
    """
    if not selected_image_name or not image_names:
        return None, None
        
    try:
        # 查找选中图片在列表中的位置
        index = image_names.index(selected_image_name)
        annotated_path = annotated_paths[index]
        mask_path = mask_paths[index]
        
        # 加载图片并返回给界面
        annotated_img = Image.open(annotated_path)
        mask_img = Image.open(mask_path)
        
        return annotated_img, mask_img
        
    except Exception as e:
        logger.error(f"Failed to load preview image: {str(e)}")
        return None, None

# =========================================================================
def process_single_video(video_path, frame_interval, stimulus_start):
    if not model or not device:
        raise RuntimeError("Model failed to initialize")

    analysis_data = []
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_name}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = tempfile.NamedTemporaryFile(suffix=f"_{video_name}.mp4", delete=False)
    out_video_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (orig_w, orig_h))

    frame_idx = 0
    mask_history = []
    max_history = 3
    palette = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (144, 238, 144),
               (0, 215, 255), (0, 165, 255), (0, 0, 255), (226, 43, 138)]

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_area = 0
        if frame_idx % frame_interval == 0:
            pil_img = preprocess_for_segmentation(frame)
            mask = segment_image(pil_img)
            mask_history.append(mask)
            if len(mask_history) > max_history: mask_history.pop(0)

        if mask_history:
            last_mask = np.mean(mask_history, axis=0).astype(np.uint8)
            last_mask = cv2.GaussianBlur(last_mask, (5, 5), 0)
            last_mask = cv2.resize(last_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(last_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                current_area = cv2.contourArea(largest_contour)
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    (xc, yc), (d1, d2), angle = ellipse
                    for i in range(8):
                        theta = i * (2 * np.pi / 8)
                        px = int(xc + (d1/2) * np.cos(theta) * np.cos(np.radians(angle)) - (d2/2) * np.sin(theta) * np.sin(np.radians(angle)))
                        py = int(yc + (d1/2) * np.cos(theta) * np.sin(np.radians(angle)) + (d2/2) * np.sin(theta) * np.cos(np.radians(angle)))
                        cv2.circle(frame, (px, py), 12, palette[i], -1)
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cv2.circle(frame, (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])), 5, (0, 0, 255), -1)

        analysis_data.append({'Frame_ID': frame_idx, 'Time_s': frame_idx / fps, 'Pupil_Area_pixels': current_area})
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # --- 核心指标计算 (恢复 Full.csv 逻辑 + Latency) ---
    df = pd.DataFrame(analysis_data)
    df['Pupil_Area_pixels'] = df['Pupil_Area_pixels'].replace(0, np.nan).interpolate().bfill()
    df['Diameter_px'] = 2 * np.sqrt(df['Pupil_Area_pixels'] / np.pi)
    df['Smooth_Area'] = df['Pupil_Area_pixels'].rolling(window=5, center=True).mean().fillna(df['Pupil_Area_pixels'])
    df['Smooth_Dia'] = 2 * np.sqrt(df['Smooth_Area'] / np.pi)

    # 1. 基线计算
    baseline_window = df[df['Time_s'] < stimulus_start]
    baseline_area = baseline_window['Smooth_Area'].mean() if not baseline_window.empty else df['Smooth_Area'].iloc[0]
    baseline_dia = 2 * np.sqrt(baseline_area / np.pi)

    post_df = df[df['Time_s'] >= stimulus_start].copy()
    metrics = {}
    
    if not post_df.empty:
        # A. 潜伏期 (Latency)
        threshold = baseline_area * 0.97
        react_df = post_df[post_df['Smooth_Area'] < threshold]
        latency = react_df.iloc[0]['Time_s'] - stimulus_start if not react_df.empty else 0

        # B. 最大收缩点 (Max Constriction)
        min_idx = post_df['Smooth_Area'].idxmin()
        min_area = post_df.loc[min_idx, 'Smooth_Area']
        min_dia = 2 * np.sqrt(min_area / np.pi)
        min_time = post_df.loc[min_idx, 'Time_s']
        constriction_time = min_time - stimulus_start

        # C. 速度指标 (Velocities)
        post_df['V_dia'] = -post_df['Smooth_Dia'].diff() * fps # 正值代表收缩
        max_con_vel = post_df[post_df['Time_s'] <= min_time]['V_dia'].max()
        avg_con_vel = (baseline_dia - min_dia) / constriction_time if constriction_time > 0 else 0
        
        # D. T75 恢复时间
        t75_target_area = min_area + 0.75 * (baseline_area - min_area)
        recovery_df = post_df[post_df['Time_s'] > min_time]
        t75_reach_df = recovery_df[recovery_df['Smooth_Area'] >= t75_target_area]
        t75_time = t75_reach_df.iloc[0]['Time_s'] - min_time if not t75_reach_df.empty else np.nan
        
        # E. 平均扩张速度
        last_time = post_df['Time_s'].iloc[-1]
        last_dia = post_df['Smooth_Dia'].iloc[-1]
        avg_dil_vel = (last_dia - min_dia) / (last_time - min_time) if last_time > min_time else 0

        metrics = {
            'Latency_s': max(0, latency),
            'Baseline_Area_px': baseline_area,
            'Max_Constriction_Value_px': min_area,
            'Constriction_Time_s': constriction_time,
            'Min_Constriction_Ratio': min_area / baseline_area if baseline_area > 0 else 0,
            'Constriction_Percent': (1 - (min_area / baseline_area)) * 100,
            'T75_Recovery_Time_s': t75_time,
            'Avg_Constriction_Vel_px_s': avg_con_vel,
            'Max_Constriction_Vel_px_s': max_con_vel,
            'Avg_Dilation_Vel_px_s': avg_dil_vel,
            'Dilatation_Ratio': last_dia / baseline_dia if baseline_dia > 0 else 0
        }
    
    df.attrs['PLR_metrics'] = metrics
    return out_video_path, df
# ... (后续生成报告的代码 generate_report_and_data 中也要确保对应的 T75 逻辑不再被调用) ...

# **======== Report and Data Generation function (Min Area and Plot return, English text) ========**
def generate_report_and_data(analysis_results, stimulus_start=0):
    """
    全量汇总分析结果：恢复所有生理指标（CV, ACV, T75等）并保留潜伏期。
    """
    if not analysis_results:
        return None, None, None, None

    combined_df_list = []
    summary_rows = [] # 用于生成每一行代表一个视频的汇总表
    
    report_summary = "================================================\n"
    report_summary += "      PLR BIOMARKERS COMPREHENSIVE REPORT       \n"
    report_summary += "================================================\n"
    
    plt.close('all')
    fig, (ax_cv, ax_cr) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    threshold_line_added = False

    for video_name, df in analysis_results:
        # 1. 获取单视频计算的所有指标
        m = df.attrs.get('PLR_metrics', {})
        
        # 2. 构建该视频的汇总行 (包含 Full.csv 中的所有指标)
        row = {'Video_Name': video_name}
        row.update(m)  # 这一步极其重要：将 metrics 字典里的所有键值对（T75, Velocity 等）全量合并进来
        summary_rows.append(row)

        # 3. 绘图与归一化逻辑 (保持百分比制)
        fps = 30 
        if 'Time_s' in df.columns and len(df) > 1:
            fps = 1 / (df['Time_s'].iloc[1] - df['Time_s'].iloc[0])

        # 使用基线面积计算直径
        base_area = m.get('Baseline_Area', df['Smooth_Area'].iloc[0])
        init_dia = 2 * np.sqrt(base_area / np.pi)
        
        df["Constriction_Ratio_Pct"] = (df['Smooth_Dia'] / init_dia) * 100
        df["Velocity_Pct_s"] = -df["Constriction_Ratio_Pct"].diff() * fps
        df["Velocity_Pct_s"] = df["Velocity_Pct_s"].rolling(window=7, center=True).mean().fillna(0)

        # 4. 绘图
        ax_cv.plot(df['Time_s'], df['Velocity_Pct_s'], label=f"{video_name}")
        ax_cr.plot(df['Time_s'], df['Constriction_Ratio_Pct'], label=f"{video_name}", linewidth=2)
        
        # --- 关键修改：添加红色垂直刺激线，移除橙色水平线 ---
        if not threshold_line_added:
            # 1. 在上方速度图 (CV) 绘制红色垂直线
            ax_cv.axvline(x=stimulus_start, color='red', linestyle='--', linewidth=1.5, label='Stimulus Onset')
            
            # 2. 在下方收缩率图 (CR) 绘制红色垂直线
            ax_cr.axvline(x=stimulus_start, color='red', linestyle='--', linewidth=1.5, label='Stimulus Onset')
            
            # 【注意】这里删除了原来的 ax_cr.axhline(y=97.0, ...) 这一行，所以橙色水平线会消失
            
            threshold_line_added = True
        
        # 5. 为了 CSV 逐帧表也包含汇总信息，合并指标列
        for key, val in m.items():
            df[key] = val
        df["Video_Source"] = video_name
        combined_df_list.append(df)

    # --- 样式修饰 ---
    ax_cv.set_title('Pupil Constriction Velocity (CV)')
    ax_cv.set_ylabel('Velocity (% / s)') 
    ax_cr.set_title('Pupil Constriction Ratio (CR)')
    ax_cr.set_ylabel('Constriction Ratio, %')
    ax_cr.axhline(100.0, color='gray', linestyle='--')
    ax_cr.set_ylim(0, 115)
    ax_cv.legend(loc='upper right', fontsize='x-small')
    ax_cr.legend(loc='upper right', fontsize='x-small')
    fig.tight_layout()

    # --- 导出全量汇总 CSV ---
    summary_df = pd.DataFrame(summary_rows)
    csv_summary_tmp = tempfile.NamedTemporaryFile(suffix="_PLR_Summary_Full.csv", delete=False)
    summary_df.to_csv(csv_summary_tmp.name, index=False)

    # --- 导出绘图 ---
    plot_img_tmp = tempfile.NamedTemporaryFile(suffix="_PLR_Plot.png", delete=False)
    fig.savefig(plot_img_tmp.name, dpi=300)
    
    # --- 生成文本报告 ---
    for r in summary_rows:
        report_summary += f"\n>>> Video: {r['Video_Name']}\n"
        for k, v in r.items():
            if k != 'Video_Name':
                report_summary += f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n"

    report_tmp = tempfile.NamedTemporaryFile(suffix="_Full_Report.txt", delete=False)
    with open(report_tmp.name, "w", encoding='utf-8') as f:
        f.write(report_summary)

    # 注意：这里返回的是包含全量指标的 summary_df 路径
    return csv_summary_tmp.name, report_tmp.name, fig, plot_img_tmp.name
# **======== Batch Video Processing function (Adjusted return values) ========**
def process_multiple_videos(video_files, frame_interval, stimulus_start):
    # 1. 在函数开头就初始化所有变量为 None
    # 这样即使后面代码报错跳过了，return 时也不会报 NameError
    zip_path = None
    csv_path = None
    report_path = None
    fig_plot = None
    processed_video_paths = []
    analysis_results = []
    first_processed_video_path = None

    if not video_files:
        return None, None, None, None, "Please upload videos.", None, None, None

    progress = gr.Progress(track_tqdm=True)

    # 2. 处理视频循环
    for i, video_file in enumerate(progress.tqdm(video_files, desc="Processing Videos")):
        try:
            out_video_path, df = process_single_video(video_file.name, frame_interval, stimulus_start)
            processed_video_paths.append(out_video_path)
            analysis_results.append((os.path.basename(video_file.name), df))
            if i == 0:
                first_processed_video_path = out_video_path
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")
            continue

    # 3. 生成报告 (这里会给 csv_path 和 report_path 赋值)
    try:
        csv_path, report_path, fig_plot, plot_img_path = generate_report_and_data(analysis_results, stimulus_start)
    except Exception as e:
        print(f"Error generating report: {e}")

    # 4. 【关键修正】定义并执行 ZIP 打包
    # 确保 zip_path 在这里被明确定义
    try:
        zip_temp_file = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        zip_path = zip_temp_file.name # <--- 在这里定义了 zip_path
        zip_temp_file.close()

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for path in processed_video_paths:
                if path and os.path.exists(path):
                    zipf.write(path, os.path.basename(path))
            if csv_path and os.path.exists(csv_path):
                zipf.write(csv_path, "PLR_Metrics_Data.csv")
            if report_path and os.path.exists(report_path):
                zipf.write(report_path, "Full_Report.txt")
    except Exception as e:
        print(f"ZIP creation error: {str(e)}")
        # 如果打包失败，zip_path 保持为 None，不会报错 NameError

    message = f"✅ Success: Processed {len(analysis_results)} videos."

    # 5. 返回结果
    return (
        first_processed_video_path, 
        csv_path, 
        report_path, 
        zip_path,  # 现在这里绝对不会报 NameError 了
        message, 
        csv_path, 
        report_path, 
        fig_plot
    )
def create_interface():

    # 📌 关键修改: CSS 样式更新
    # 1. 标题居中
    # 2. Primary Button Color: #8A2BE2 (Violet/Electric Purple)
    # 3. General theme adjustments
    css_theme = """
    .gradio-container {
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    h1 {
        color: #004080; 
        font-weight: 800 !important;      /* 800 是超粗体，让标题更有力量感 */
        text-align: center;
        font-size: 48px !important;       /* 👈 这里控制大小，48px 通常是描述文字的 2.5 倍大 */
        margin-top: 10px;                 /* 调整顶部间距 */
        margin-bottom: 0px;               /* 缩短与下方描述文字的距离 */
    }
    .panel {
        background-color: #f7f7f7; 
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }
    .primary_btn {
        background-color: #8A2BE2 !important; /* 📌 Key change: Violet/Purple */
        border-color: #8A2BE2 !important;
        color: white !important;
        transition: background-color 0.2s;
    }
    .primary_btn:hover {
        background-color: #7A1DE1 !important; /* Slightly darker on hover */
    }
    .secondary_btn {
        background-color: #6c757d !important; 
        border-color: #6c757d !important;
        color: white !important;
    }
    """


    with gr.Blocks(title="PupilAI Analysis Platform", css=css_theme) as demo:

        # 📌 语言修改: 标题文本改为英文
        gr.Markdown(
            """
            # PupilAI Analysis Platform
            **Advanced Mouse Pupil Segmentation and Metric Extraction System for Video and Static Image Analysis.**
            ---
            """
        )
        
        # State variables (No change in logic)
        image_names_state = gr.State(value=[])
        annotated_paths_state = gr.State(value=[])
        mask_paths_state = gr.State(value=[])

        
        with gr.Tabs():

            # ==========================================================
            # **TAB 1: Video Batch Analysis (Tab and all components in English)** # ==========================================================
            with gr.TabItem("🎥 Video Batch Analysis"):
                with gr.Row(equal_height=False):

                    # === Left: Input, Preview, and Config ===
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("## 📁 Data and Preprocessing", elem_classes="panel")

                        # --- Video Input ---
                        with gr.Group():
                            gr.Markdown("### 🎥 Input Video Files")
                            # 📌 语言修改
                            video_input = gr.File(
                                label="Upload Experiment Videos (Supports MP4, AVI, MOV, MKV, WEBM)",
                                file_types=[
                                    ".mp4", ".MP4", 
                                    ".avi", ".AVI", 
                                    ".mov", ".MOV", 
                                    ".mkv", ".MKV", 
                                    ".webm", ".WEBM"
                                ],
                                file_count="multiple",
                                type="filepath"
                            )

                        # --- Image Preprocessing Parameters (Placeholders) ---
                        with gr.Group(elem_classes="panel"):
                            gr.Markdown("### ✨ Preprocessing Parameters")
                            # 📌 语言修改
                            gr.Slider(1.0, 3.0, value=1.00, step=0.01, label="Contrast")
                            gr.Slider(-1.0, 1.0, value=0.00, step=0.01, label="Brightness")
                            gr.Slider(0.5, 2.0, value=1.00, step=0.01, label="Gamma")
                            gr.Checkbox(label="Invert Color", value=False)
                            gr.Button("Reset Parameters", variant="secondary", size="sm")

                        # --- Core Control and Model Parameters --- 📌 迁移至此
                        with gr.Group(elem_classes="panel"):
                            gr.Markdown("### ⚙️ Core Control and Configuration")
                            # Core Parameter
                            # 📌 语言修改
                            frame_slider = gr.Slider(
                                1, 30, value=5, step=1,
                                label="**Segmentation Frame Interval**",
                                info="Segment every N frames for balance between speed and accuracy."
                            )
                            stimulus_input = gr.Number(
                            value=5,
                            label="Stimulus Start Time (seconds)",
                            info="Set the light stimulus onset time (e.g., 5 means light turns on at 5s)"
                            )

                            # Run Button (Primary color changed via CSS)
                            # 📌 语言修改
                            run_button = gr.Button("▶️ Run Batch Analysis", variant="primary", size="lg")


                    # === Middle: Main Output and Control ===
                    with gr.Column(scale=2, min_width=600):
                        gr.Markdown("## 📺 Video Processing Output", elem_classes="panel")

                        # --- Video Preview Area ---
                        # 📌 语言修改
                        processed_video_preview = gr.Video(label="Preview of First Processed Video (with Pupil Annotation)", format="mp4")
                        # 📌 语言修改
                        analysis_message = gr.Markdown("🚀 **Waiting for analysis to start**...", elem_id="analysis_msg")
                        
                        # --- Metric Plot Area --- 📌 迁移至此
                        with gr.Group(elem_classes="panel"): # 确保迁移后的 Group 样式正确
                            gr.Markdown("### 📈 Metric Trend Plot")
                            # 📌 语言修改
                            video_plot_output = gr.Plot(label="Pupil Area Change Over Time", visible=True) 


                    # === Right: Data and Reports ===
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("## 📊 Reports and Downloads", elem_classes="panel")

                        # --- Report Download Area ---
                        with gr.Group():
                            gr.Markdown("### ⬇️ Data Report Downloads")
                            # 📌 语言修改
                            download_csv = gr.File(label="CSV - Raw Metric Data", file_types=[".csv"])
                            download_report = gr.File(label="TXT/PDF - Summary Report", file_types=[".txt", ".pdf"])
                            download_zip = gr.File(label="ZIP - All Annotated Videos", file_types=[".zip"])
                    # Binding events (No logic change)
                    run_button.click(
                        fn=process_multiple_videos,
                        inputs=[video_input, frame_slider, stimulus_input],
                        outputs=[
                            processed_video_preview,
                            download_csv,
                            download_report,
                            download_zip,
                            analysis_message,
                            download_csv, 
                            download_report,
                            video_plot_output 
                        ],
                        api_name="process_batch_videos"
                    )

            

            # ==========================================================
            # **TAB 2: Image/Batch Static Analysis (Tab and all components in English)** # ==========================================================
            with gr.TabItem("🖼️ Static Image Batch Analysis"):
                with gr.Row(equal_height=False):
                    
                    # === Left: Image Input and Control ===
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("## 📁 Batch Image Input", elem_classes="panel")
                        
                        # 📌 语言修改
                        image_input = gr.File(
                            label="Upload Images (Supports JPG, PNG, JPEG, BMP, WEBP)",
                            file_types=[
                                ".jpg", ".JPG",
                                ".jpeg", ".JPEG",
                                ".png", ".PNG",
                                ".bmp", ".BMP",
                                ".webp", ".WEBP"
                            ],
                            file_count="multiple", 
                            type="filepath"
                        )
                        # 📌 语言修改 (Primary color changed via CSS)
                        image_run_button = gr.Button("▶️ Run Batch Segmentation", variant="primary", size="lg")
                        image_status_msg = gr.Markdown("🚀 **Waiting for image upload**...")

                    # === Middle/Right: Results Output ===
                    with gr.Column(scale=2, min_width=600):
                        gr.Markdown("## 💡 Results Preview and Downloads", elem_classes="panel")
                        
                        # --- Preview Selector ---
                        # 📌 语言修改
                        image_selector = gr.Dropdown(
                            label="Select Image for Preview",
                            choices=[], 
                            value=None,
                            interactive=True
                        )
                        
                        # --- Result Preview (Display selected image) ---
                        with gr.Row():
                            # 📌 语言修改
                            image_output = gr.Image(label="Preview: Fitted Ellipse Result (Cyan)", image_mode="RGB", scale=1)
                            ellipse_mask_output = gr.Image(label="Preview: Fitted Ellipse Mask (Binary)", image_mode="RGB", scale=1)
                        
                        # --- Report Download Area ---
                        gr.Markdown("### ⬇️ Batch Results Downloads")
                        # 📌 语言修改
                        download_image_csv = gr.File(label="CSV - Merged Metric Data for all Images", file_types=[".csv"])
                        download_image_zip = gr.File(label="ZIP - All Annotated and Mask Images", file_types=[".zip"])
                        
                        # --- Merged Quantitative Metrics Table Preview ---
                        gr.Markdown("### 📈 Merged Metric Preview")
                        # 📌 语言修改
                        image_metrics_table = gr.Dataframe(
                            headers=['Image_Name', 'Pupil_Area_pixels', 'Equivalent_Diameter_pixels', 'Aspect_Ratio', 'Status'], 
                            row_count=(5, 'dynamic'), 
                            col_count=(5, 'fixed'), 
                            interactive=False,
                            value=pd.DataFrame(columns=['Image_Name', 'Pupil_Area_pixels', 'Equivalent_Diameter_pixels', 'Aspect_Ratio', 'Status']) 
                        )
                    
                    # **[Binding 1] Run Batch Segmentation Event** (No logic change)
                    run_output_tuple = image_run_button.click(
                        fn=process_multiple_images,
                        inputs=[image_input],
                        outputs=[
                            image_names_state,        
                            annotated_paths_state,    
                            mask_paths_state,         
                            image_selector,           
                            download_image_csv,       
                            download_image_zip,       
                            image_status_msg,         
                            image_metrics_table       
                        ],
                        api_name="process_batch_images"
                    )
                    
                    # **[Binding 2] Update image preview on selector change** (No logic change)
                    image_selector.change(
                        fn=update_image_preview,
                        inputs=[
                            image_selector, 
                            image_names_state, 
                            annotated_paths_state, 
                            mask_paths_state
                        ],
                        outputs=[image_output, ellipse_mask_output],
                    )
                    
                    # **[Binding 3] Force preview update after batch processing** (No logic change)
                    run_output_tuple.then(
                        fn=update_image_preview,
                        inputs=[
                            image_selector, 
                            image_names_state, 
                            annotated_paths_state, 
                            mask_paths_state
                        ],
                        outputs=[image_output, ellipse_mask_output]
                    )

    return demo


# ===== Application startup (No change) =====
if __name__ == "__main__":
    try:
        logger.info("Starting Gradio application...")
        iface = create_interface()
        iface.launch(mcp_server=False)
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
