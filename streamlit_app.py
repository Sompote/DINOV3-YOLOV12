#!/usr/bin/env python3
"""
YOLOv12-DINO Streamlit Object Detection App

A Streamlit-based web interface for YOLOv12-DINO object detection.
Upload images and get real-time object detection results with interactive controls.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
import sys
from typing import Dict, List, Tuple
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from inference import YOLOInference
except ImportError as e:
    st.error(f"Error importing inference module: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="YOLOv12-DINO Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure large file uploads
import streamlit.web.bootstrap
try:
    # Set large upload size (5GB in MB)
    if hasattr(st, '_config'):
        st._config.set_option('server.maxUploadSize', 5000)
    else:
        # For newer Streamlit versions
        from streamlit import config
        config.set_option('server.maxUploadSize', 5000)
        config.set_option('server.maxMessageSize', 5000)
except Exception:
    pass

# Set environment variable for large files (5GB in MB)
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '5000'

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .detection-box {
        border: 2px solid #1e88e5;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .success-box {
        border: 2px solid #4caf50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f1f8e9;
    }
    .error-box {
        border: 2px solid #f44336;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #ffebee;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_instance' not in st.session_state:
    st.session_state.model_instance = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

def load_model(weights_path: str, device: str = "cpu") -> Tuple[bool, str]:
    """Load the YOLO model with specified weights."""
    try:
        if not Path(weights_path).exists():
            return False, f"❌ Model file not found: {weights_path}"
        
        with st.spinner("Loading YOLOv12-DINO model..."):
            st.session_state.model_instance = YOLOInference(
                weights=weights_path,
                conf=0.25,
                iou=0.7,
                imgsz=640,
                device=device,
                verbose=True
            )
        
        st.session_state.model_loaded = True
        
        # Get model information
        model_info = f"✅ Model loaded successfully\n"
        model_info += f"📋 Task: {st.session_state.model_instance.model.task}\n"
        
        if hasattr(st.session_state.model_instance.model.model, 'names'):
            class_names = list(st.session_state.model_instance.model.model.names.values())
            model_info += f"🏷️ Classes ({len(class_names)}): {', '.join(class_names)}"
            
        return True, model_info
        
    except Exception as e:
        return False, f"❌ Error loading model: {str(e)}"

def perform_detection(
    image: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    image_size: int
) -> Tuple[np.ndarray, Dict, str]:
    """Perform object detection using the same method as inference.py."""
    
    if st.session_state.model_instance is None:
        return None, {}, "❌ No model loaded"
    
    try:
        # Update model parameters
        st.session_state.model_instance.conf = conf_threshold
        st.session_state.model_instance.iou = iou_threshold
        st.session_state.model_instance.imgsz = image_size
        
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Convert RGB to BGR for OpenCV (same as inference.py)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_file.name, image_bgr)
            tmp_path = tmp_file.name
        
        start_time = time.time()
        
        try:
            # Use the exact same method as inference.py
            results = st.session_state.model_instance.predict_single(
                source=tmp_path,
                save=False,
                show=False,
                save_txt=False,
                save_conf=False,
                save_crop=False,
                output_dir=None
            )
            
            inference_time = time.time() - start_time
            
            if not results:
                return image, {}, "❌ No results returned from model"
            
            result = results[0]
            
            # Get annotated image
            annotated_img = result.plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Process detection results
            detection_data = process_detection_results(result, inference_time)
            
            # Generate summary text
            summary_text = generate_detection_summary(result, detection_data, inference_time)
            
            return annotated_img, detection_data, summary_text
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        return None, {}, f"❌ Error during detection: {str(e)}"

def process_detection_results(result, inference_time: float) -> Dict:
    """Process detection results into structured data."""
    
    if result.boxes is None or len(result.boxes) == 0:
        return {
            'total_detections': 0,
            'class_counts': {},
            'detections': [],
            'inference_time': inference_time
        }
    
    detections = result.boxes
    
    # Get class names
    if hasattr(st.session_state.model_instance.model.model, 'names'):
        class_names = st.session_state.model_instance.model.model.names
    else:
        class_names = getattr(result, 'names', {i: f"Class_{i}" for i in range(100)})
    
    # Process each detection
    detection_list = []
    class_counts = {}
    
    for i, (box, conf, cls) in enumerate(zip(detections.xyxy, detections.conf, detections.cls)):
        cls_id = int(cls)
        cls_name = class_names.get(cls_id, f"Class_{cls_id}")
        confidence = float(conf)
        
        x1, y1, x2, y2 = box.tolist()
        
        detection_list.append({
            'id': i + 1,
            'class': cls_name,
            'confidence': confidence,
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2),
            'width': int(x2 - x1),
            'height': int(y2 - y1),
            'area': int((x2 - x1) * (y2 - y1))
        })
        
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    return {
        'total_detections': len(detection_list),
        'class_counts': class_counts,
        'detections': detection_list,
        'inference_time': inference_time
    }

def generate_detection_summary(result, detection_data: Dict, inference_time: float) -> str:
    """Generate detection summary text (same format as inference.py)."""
    
    total_detections = detection_data['total_detections']
    
    if total_detections == 0:
        return "🔍 No objects detected in the image."
    
    summary = f"✅ **Detection Results:**\n\n"
    summary += f"📊 **Images processed:** 1\n"
    summary += f"📊 **Total detections:** {total_detections}\n"
    summary += f"⏱️ **Inference time:** {inference_time:.3f}s\n\n"
    
    summary += "📋 **Detections by class:**\n"
    for cls_name, count in sorted(detection_data['class_counts'].items()):
        summary += f"   • {cls_name}: {count}\n"
    
    return summary

def create_detection_chart(detection_data: Dict):
    """Create interactive charts for detection results."""
    
    if detection_data['total_detections'] == 0:
        st.info("No detections to visualize")
        return
    
    # Class distribution pie chart
    class_counts = detection_data['class_counts']
    
    fig_pie = px.pie(
        values=list(class_counts.values()),
        names=list(class_counts.keys()),
        title="Detection Distribution by Class"
    )
    fig_pie.update_layout(height=400)
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Confidence distribution
    confidences = [det['confidence'] for det in detection_data['detections']]
    classes = [det['class'] for det in detection_data['detections']]
    
    fig_conf = px.box(
        x=classes,
        y=confidences,
        title="Confidence Distribution by Class"
    )
    fig_conf.update_layout(height=400)
    fig_conf.update_xaxes(title="Class")
    fig_conf.update_yaxes(title="Confidence Score")
    
    st.plotly_chart(fig_conf, use_container_width=True)

def create_detection_table(detection_data: Dict):
    """Create detailed detection table."""
    
    if detection_data['total_detections'] == 0:
        st.info("No detections to display")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(detection_data['detections'])
    
    # Format confidence as percentage
    df['confidence_pct'] = df['confidence'].apply(lambda x: f"{x:.1%}")
    
    # Reorder columns for better display
    display_columns = ['id', 'class', 'confidence_pct', 'x1', 'y1', 'x2', 'y2', 'width', 'height', 'area']
    df_display = df[display_columns].copy()
    
    # Rename columns for better readability
    df_display.columns = ['ID', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2', 'Width', 'Height', 'Area']
    
    st.dataframe(df_display, use_container_width=True)
    
    # Download button for results
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Download Detection Results (CSV)",
        data=csv,
        file_name=f"detection_results_{int(time.time())}.csv",
        mime="text/csv"
    )

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 YOLOv12-DINO Object Detection</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive object detection powered by YOLOv12 + DINOv3 Vision Transformers**")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("🛠️ Model Configuration")
        
        # Model loading section
        st.subheader("📁 Load Model")
        st.info("💡 **Large file support** - Upload YOLOv12-DINO models up to 5GB")
        uploaded_file = st.file_uploader(
            "Upload Model Weights (.pt file)",
            type=['pt'],
            help="Upload your trained YOLOv12-DINO model weights (no size limit)",
            label_visibility="visible"
        )
        
        device = st.selectbox(
            "Device",
            options=["cpu", "cuda", "mps"],
            index=0,
            help="Select computation device"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.success(f"📁 File uploaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")
            
            if st.button("🔄 Load Model", type="primary"):
                # Show progress for large file processing
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Save uploaded file temporarily with progress indication
                    status_text.text("💾 Saving uploaded file...")
                    progress_bar.progress(25)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    progress_bar.progress(50)
                    status_text.text("🚀 Loading model...")
                    
                    success, message = load_model(tmp_path, device)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Model loading complete!")
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                        
                    # Clean up progress indicators
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
        
        # Detection parameters
        st.subheader("⚙️ Detection Parameters")
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.01,
            max_value=1.0,
            value=0.25,
            step=0.01,
            help="Minimum confidence for detection"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.01,
            max_value=1.0,
            value=0.7,
            step=0.01,
            help="IoU threshold for Non-Maximum Suppression"
        )
        
        image_size = st.selectbox(
            "Image Size",
            options=[320, 416, 512, 640, 832, 1024, 1280],
            index=3,
            help="Input image size for model"
        )
        
        # Model status
        if st.session_state.model_loaded:
            st.success("✅ Model loaded and ready")
        else:
            st.warning("⚠️ Please load a model first")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🖼️ Input Image")
        
        # Image upload
        image_file = st.file_uploader(
            "Upload Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image for object detection"
        )
        
        if image_file is not None:
            # Load and display image
            try:
                image = Image.open(image_file)
                image_np = np.array(image)
                
                # Display image with compatibility for different Streamlit versions
                st.image(image, caption="Input Image", width=None)
                
                # Image info
                st.info(f"📊 Image size: {image.width} × {image.height} pixels")
                
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                return
            
            # Detection button
            if st.button("🔍 Detect Objects", type="primary", disabled=not st.session_state.model_loaded):
                if st.session_state.model_loaded:
                    with st.spinner("Performing object detection..."):
                        annotated_img, detection_data, summary_text = perform_detection(
                            image_np, conf_threshold, iou_threshold, image_size
                        )
                        
                        # Store results in session state
                        st.session_state.last_detection = {
                            'annotated_img': annotated_img,
                            'detection_data': detection_data,
                            'summary_text': summary_text,
                            'timestamp': time.time()
                        }
                        
                        # Add to history
                        st.session_state.detection_history.append({
                            'filename': image_file.name,
                            'detections': detection_data['total_detections'],
                            'timestamp': time.time()
                        })
                else:
                    st.error("❌ Please load a model first")
    
    with col2:
        st.header("🎯 Detection Results")
        
        if 'last_detection' in st.session_state:
            detection_result = st.session_state.last_detection
            
            # Display annotated image
            if detection_result['annotated_img'] is not None:
                try:
                    st.image(
                        detection_result['annotated_img'],
                        caption="Detection Results",
                        width=None
                    )
                except Exception as e:
                    st.error(f"Error displaying result image: {str(e)}")
            
            # Display summary
            st.markdown(f'<div class="detection-box">{detection_result["summary_text"]}</div>', 
                       unsafe_allow_html=True)
            
        else:
            st.info("🔍 Upload an image and click 'Detect Objects' to see results")
    
    # Additional tabs for detailed analysis
    if 'last_detection' in st.session_state and st.session_state.last_detection['detection_data']['total_detections'] > 0:
        
        st.header("📊 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📈 Charts", "📋 Detection Table", "📜 History"])
        
        with tab1:
            create_detection_chart(st.session_state.last_detection['detection_data'])
        
        with tab2:
            create_detection_table(st.session_state.last_detection['detection_data'])
        
        with tab3:
            if st.session_state.detection_history:
                history_df = pd.DataFrame(st.session_state.detection_history)
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], unit='s')
                history_df.columns = ['Filename', 'Detections', 'Timestamp']
                
                st.dataframe(history_df, use_container_width=True)
                
                # Clear history button
                if st.button("🗑️ Clear History"):
                    st.session_state.detection_history = []
                    st.rerun()
            else:
                st.info("No detection history yet")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            🚀 <strong>YOLOv12-DINO Object Detection</strong> | 
            Powered by <strong>Streamlit</strong> + <strong>YOLOv12</strong> + <strong>DINOv3</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()