"""
Phát hiện hành vi bạo lực — Streamlit + YOLO (best.pt).
Chạy: streamlit run app.py
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS_PT = ROOT / "best.pt"
# WEIGHTS_ONNX = ROOT / "demo" / "best.onnx"


def resolve_model_path() -> Path:
    if WEIGHTS_PT.exists():
        return WEIGHTS_PT
    # if WEIGHTS_ONNX.exists():
    #     return WEIGHTS_ONNX
    # raise FileNotFoundError("Không tìm thấy best.pt hoặc best.onnx trong thư mục demo.")


def label_vi(raw: str) -> str:
    """Chuyển đổi tên class sang tiếng Việt"""
    display_names = {
        "binh thuong": "Bình thường",
        "hanh vi bao luc": "Hành vi bạo lực",
    }
    return display_names.get(raw, raw)


def boxes_to_labels(boxes, names: dict) -> list[str]:
    if boxes is None or len(boxes) == 0:
        return []
    return [label_vi(names[int(c)]) for c in boxes.cls.int().tolist()]


# =============================
# 1. Load YOLO model
# =============================
@st.cache_resource
def load_model():
    return YOLO(str(resolve_model_path()), task="detect")

model = load_model()
CLASS_NAMES = model.names


# =============================
# 2. Streamlit UI
# =============================
st.set_page_config(
    page_title="Phát hiện hành vi bạo lực",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .stTitle {
        color: white !important;
        font-weight: 700;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .result-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .danger-high {
        background: #fee;
        border-left-color: #dc3545;
    }
    .danger-medium {
        background: #fff3cd;
        border-left-color: #ffc107;
    }
    .danger-low {
        background: #d4edda;
        border-left-color: #28a745;
    }
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem;">🛡️ Hệ thống Nhận diện Hành vi Bạo lực</h1>
    <p style="margin: 1rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
        Sử dụng YOLO11n - Mô hình AI tiên tiến cho phát hiện thời gian thực
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<div style="padding: 1rem 0; border-bottom: 2px solid #667eea; margin-bottom: 1rem;">
    <h2 style="margin: 0; color: #667eea;">⚙️ Cài đặt</h2>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎯 Ngưỡng Confidence")
    conf_threshold = st.slider(
        "Độ tin cậy tối thiểu",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Giá trị confidence thấp hơn sẽ phát hiện nhiều đối tượng hơn nhưng có thể kém chính xác hơn"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Thông tin")
    st.info("""
    **Model:** YOLO11n
    **Classes:** 2 (Bình thường, Bạo lực)
    **Input:** Ảnh/Video
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Mức độ nguy hiểm")
    st.markdown("""
    <div style="padding: 0.5rem; background: #fee; border-left: 4px solid #dc3545; border-radius: 5px; margin: 0.5rem 0;">
        🔴 <strong>NGUY HIỂM:</strong> ≥ 2 hành vi bạo lực
    </div>
    <div style="padding: 0.5rem; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 5px; margin: 0.5rem 0;">
        🟡 <strong>CẢNH BÁO:</strong> 1 hành vi bạo lực
    </div>
    <div style="padding: 0.5rem; background: #d4edda; border-left: 4px solid #28a745; border-radius: 5px; margin: 0.5rem 0;">
        🟢 <strong>BÌNH THƯỜNG:</strong> Không có bạo lực
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📷 Ảnh", "🎬 Video"])

# =============================
# Tab Ảnh
# =============================
with tab1:
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 10px; margin-bottom: 1.5rem;">
        <h2 style="margin: 0; color: #667eea;">📷 Phân tích Ảnh</h2>
        <p style="margin: 0.5rem 0 0 0; color: #666;">Tải lên một hoặc nhiều hình ảnh để phát hiện hành vi bạo lực</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📂 Chọn ảnh để tải lên",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="Hỗ trợ định dạng: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_files:
        results = []
        total_objects = 0
        all_labels = []
        
        for uploaded_file in uploaded_files:
            # Đọc ảnh
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error(f"Không thể đọc file: {uploaded_file.name}")
                continue
            
            # Detect
            detection = model(img, conf=conf_threshold, verbose=False)
            boxes = detection[0].boxes
            num_objects = 0 if boxes is None else len(boxes)
            total_objects += num_objects
            
            # Lấy labels
            labels = boxes_to_labels(boxes, CLASS_NAMES)
            all_labels.extend(labels)
            
            if num_objects > 0:
                max_conf = float(boxes.conf.max())
                status = "✅ Phát hiện đối tượng"
                
                label_counts = Counter(labels)
                violence_count = label_counts.get("Hành vi bạo lực", 0)
                normal_count = label_counts.get("Bình thường", 0)
                
                if violence_count > 0:
                    danger_level = "🔴 NGUY HIỂM" if violence_count >= 2 else "🟡 CẢNH BÁO"
                else:
                    danger_level = "🟢 BÌNH THƯỜNG"
            else:
                max_conf = 0.0
                status = "❌ Không phát hiện"
                violence_count = 0
                normal_count = 0
                danger_level = "-"
            
            # Vẽ bounding boxes
            annotated = detection[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            results.append({
                "filename": uploaded_file.name,
                "status": status,
                "num_objects": num_objects,
                "max_conf": round(max_conf, 3),
                "danger_level": danger_level,
                "violence_count": violence_count,
                "normal_count": normal_count,
                "labels": labels,
                "image": annotated_rgb
            })
        
        # Hiển thị kết quả
        st.markdown("## 📊 Kết quả Phân tích")
        
        for idx, result in enumerate(results, 1):
            # Determine danger class for styling
            danger_class = "danger-high" if result['violence_count'] >= 2 else "danger-medium" if result['violence_count'] > 0 else "danger-low"
            
            st.markdown(f"""
            <div class="result-container {danger_class}">
                <h3 style="margin: 0 0 1rem 0; color: #333;">📄 {result['filename']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(result["image"], caption=f"Ảnh {idx}", width="stretch")
            
            with col2:
                # Status badge
                status_color = "#28a745" if "✅" in result['status'] else "#dc3545"
                st.markdown(f"""
                <div style="padding: 0.75rem; background: {status_color}15; border-radius: 8px; margin-bottom: 1rem;">
                    <strong style="color: {status_color};">Trạng thái:</strong> {result['status']}
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics in a grid
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Số đối tượng", result['num_objects'])
                    st.metric("Confidence max", f"{result['max_conf']:.3f}")
                with metric_col2:
                    st.metric("Hành vi bạo lực", result['violence_count'])
                    st.metric("Bình thường", result['normal_count'])
                
                # Danger level badge
                danger_bg = "#dc3545" if result['violence_count'] >= 2 else "#ffc107" if result['violence_count'] > 0 else "#28a745"
                st.markdown(f"""
                <div style="padding: 0.75rem; background: {danger_bg}; color: white; border-radius: 8px; text-align: center; font-weight: bold; margin: 1rem 0;">
                    {result['danger_level']}
                </div>
                """, unsafe_allow_html=True)
                
                # Labels
                if result['labels']:
                    st.markdown("**Phân loại:**")
                    for label in result['labels']:
                        label_color = "#dc3545" if "bạo lực" in label.lower() else "#28a745"
                        st.markdown(f"""
                        <span style="display: inline-block; padding: 0.25rem 0.75rem; background: {label_color}15; color: {label_color}; border-radius: 15px; margin: 0.25rem; font-size: 0.9rem;">
                            {label}
                        </span>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Không phát hiện đối tượng nào")
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Tổng hợp
        label_counts = Counter(all_labels)
        top_objects = ", ".join(f"{name} ({n})" for name, n in label_counts.most_common())
        if not top_objects:
            top_objects = "Không có"
        
        total_violence = label_counts.get("Hành vi bạo lực", 0)
        total_normal = label_counts.get("Bình thường", 0)
        
        st.markdown("---")
        st.markdown("## 📈 Tổng Kết")
        
        # Summary card
        summary_bg = "#dc3545" if total_violence >= 2 else "#ffc107" if total_violence > 0 else "#28a745"
        st.markdown(f"""
        <div style="padding: 1.5rem; background: {summary_bg}; color: white; border-radius: 10px; text-align: center; margin-bottom: 1.5rem;">
            <h2 style="margin: 0; font-size: 1.5rem;">
                {'🔴 CẢNH BÁO: Phát hiện nhiều hành vi bạo lực!' if total_violence >= 2 else '🟡 Cảnh báo: Phát hiện hành vi bạo lực' if total_violence > 0 else '🟢 An toàn: Không phát hiện bạo lực'}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics grid
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📷 Tổng số ảnh", len(uploaded_files))
        with col2:
            st.metric("🎯 Tổng đối tượng", total_objects)
        with col3:
            st.metric("⚠️ Hành vi bạo lực", total_violence, delta_color="inverse" if total_violence > 0 else "normal")
        with col4:
            st.metric("✅ Bình thường", total_normal)
        
        st.markdown(f"""
        <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; margin-top: 1rem;">
            <strong>Phân loại chi tiết:</strong> {top_objects}
        </div>
        """, unsafe_allow_html=True)

# =============================
# Tab Video
# =============================
with tab2:
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 10px; margin-bottom: 1.5rem;">
        <h2 style="margin: 0; color: #667eea;">🎬 Phân tích Video</h2>
        <p style="margin: 0.5rem 0 0 0; color: #666;">Tải lên video để phát hiện hành vi bạo lực theo thời gian thực</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "📂 Chọn video để tải lên",
        type=["mp4", "avi", "mov", "mkv"],
        help="Hỗ trợ định dạng: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_video:
        # Lưu video tạm
        temp_path = Path(tempfile.gettempdir()) / f"temp_{uploaded_video.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            st.error("Không mở được file video")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            
            st.markdown(f"""
            <div style="padding: 1rem; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 8px; margin: 1rem 0;">
                <strong>📹 Thông tin video:</strong> {w}x{h} | {fps} FPS | {total_frames} frames
            </div>
            """, unsafe_allow_html=True)
            
            # Nút bắt đầu phân tích
            if st.button("🔍 Bắt đầu phân tích video", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                import imageio
                out_path = Path(tempfile.gettempdir()) / f"output_{Path(uploaded_video.name).stem}_labeled.mp4"
                writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
                
                object_counter: Counter = Counter()
                frames_with_objects = 0
                frames_with_violence = 0
                frame_idx = 0
                max_conf_video = 0.0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = model(frame, conf=conf_threshold, verbose=False)
                    boxes = results[0].boxes
                    annotated = results[0].plot()
                    writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                    
                    if boxes is not None and len(boxes) > 0:
                        frames_with_objects += 1
                        max_conf_video = max(max_conf_video, float(boxes.conf.max()))
                        
                        for raw in [CLASS_NAMES[int(c)] for c in boxes.cls.int().tolist()]:
                            vi_name = label_vi(raw)
                            object_counter[vi_name] += 1
                            if vi_name == "Hành vi bạo lực":
                                frames_with_violence += 1
                    
                    frame_idx += 1
                    
                    # Cập nhật progress
                    if total_frames > 0:
                        progress = frame_idx / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Đang xử lý frame {frame_idx}/{total_frames}")
                
                cap.release()
                writer.close()

                try:
                    temp_path.unlink(missing_ok=True)
                except PermissionError:
                    pass

                progress_bar.progress(1.0)
                status_text.text("✅ Hoàn thành!")
                
                # Hiển thị kết quả
                st.markdown("## 📊 Kết quả Phân tích Video")
                
                # Summary card
                violence_ratio = (frames_with_violence / frame_idx * 100) if frame_idx > 0 else 0
                summary_bg = "#dc3545" if violence_ratio > 10 else "#ffc107" if violence_ratio > 0 else "#28a745"
                st.markdown(f"""
                <div style="padding: 1.5rem; background: {summary_bg}; color: white; border-radius: 10px; text-align: center; margin-bottom: 1.5rem;">
                    <h2 style="margin: 0; font-size: 1.5rem;">
                        {'🔴 NGUY HIỂM: Video có nhiều hành vi bạo lực!' if violence_ratio > 10 else '🟡 Cảnh báo: Video có hành vi bạo lực' if violence_ratio > 0 else '🟢 An toàn: Video không có bạo lực'}
                    </h2>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                        {violence_ratio:.1f}% frames có hành vi bạo lực
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics grid
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎬 Tổng frame", frame_idx)
                with col2:
                    st.metric("🎯 Frame có đối tượng", frames_with_objects)
                with col3:
                    st.metric("⚠️ Frame có bạo lực", frames_with_violence)
                with col4:
                    st.metric("📊 Loại đối tượng", len(object_counter))
                
                # Confidence metric
                st.markdown(f"""
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; margin: 1rem 0; text-align: center;">
                    <strong>Confidence cao nhất:</strong> {max_conf_video:.3f}
                </div>
                """, unsafe_allow_html=True)
                
                # Bảng thống kê
                st.markdown("### 📈 Thống kê chi tiết")
                if object_counter:
                    for name, count in object_counter.most_common():
                        label_color = "#dc3545" if "bạo lực" in name.lower() else "#28a745"
                        st.markdown(f"""
                        <div style="padding: 0.75rem; background: {label_color}15; border-left: 4px solid {label_color}; border-radius: 5px; margin: 0.5rem 0; display: flex; justify-content: space-between; align-items: center;">
                            <strong style="color: {label_color};">{name}</strong>
                            <span style="font-size: 1.2rem; font-weight: bold;">{count}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Không phát hiện đối tượng nào")
                # Hiển thị video trên web
                st.markdown("### 🎥 Xem Video Kết Quả")
                with open(out_path, "rb") as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
                # Download video output
                st.markdown("---")
                st.markdown("### 📥 Tải Video Kết Quả")
                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇️ Tải video đã gắn nhãn",
                        f,
                        file_name=f"output_{Path(uploaded_video.name).stem}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                
