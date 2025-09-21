#!/usr/bin/env python3
"""
Simple YOLOv12-DINO Streamlit Object Detection App
A simplified version with better compatibility across Streamlit versions.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
import sys
import time
from PIL import Image

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
    layout="wide"
)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_instance' not in st.session_state:
    st.session_state.model_instance = None

def load_model(weights_path: str, device: str = "cpu"):
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

def perform_detection(image_np, conf_threshold, iou_threshold, image_size):
    """Perform object detection using the same method as inference.py."""
    
    if st.session_state.model_instance is None:
        return None, "❌ No model loaded"
    
    try:
        # Update model parameters
        st.session_state.model_instance.conf = conf_threshold
        st.session_state.model_instance.iou = iou_threshold
        st.session_state.model_instance.imgsz = image_size
        
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Convert RGB to BGR for OpenCV (same as inference.py)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_file.name, image_bgr)
            tmp_path = tmp_file.name
        
        start_time = time.time()
        
        try:
            # Use the exact same method as inference.py
            results = st.session_state.model_instance.predict_single(
                source=tmp_path,
                save=False,
                show=False
            )
            
            inference_time = time.time() - start_time
            
            if not results:
                return image_np, "❌ No results returned from model"
            
            result = results[0]
            
            # Get annotated image
            annotated_img = result.plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Generate summary text
            summary_text = generate_summary(result, inference_time)
            
            return annotated_img, summary_text
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        return None, f"❌ Error during detection: {str(e)}"

def generate_summary(result, inference_time):
    """Generate detection summary text."""
    
    if result.boxes is None or len(result.boxes) == 0:
        return "🔍 No objects detected in the image."
    
    detections = result.boxes
    total_detections = len(detections)
    
    # Get class names
    if hasattr(st.session_state.model_instance.model.model, 'names'):
        class_names = st.session_state.model_instance.model.model.names
    else:
        class_names = getattr(result, 'names', {i: f"Class_{i}" for i in range(100)})
    
    # Count detections by class
    class_counts = {}
    for box, conf, cls in zip(detections.xyxy, detections.conf, detections.cls):
        cls_id = int(cls)
        cls_name = class_names.get(cls_id, f"Class_{cls_id}")
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    # Generate summary
    summary = f"✅ **Detection Results:**\n\n"
    summary += f"📊 **Images processed:** 1\n"
    summary += f"📊 **Total detections:** {total_detections}\n"
    summary += f"⏱️ **Inference time:** {inference_time:.3f}s\n\n"
    
    summary += "📋 **Detections by class:**\n"
    for cls_name, count in sorted(class_counts.items()):
        summary += f"   • {cls_name}: {count}\n"
    
    return summary

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🎯 YOLOv12-DINO Object Detection")
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
            help="Upload your trained YOLOv12-DINO model weights"
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
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                success, message = load_model(tmp_path, device)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Detection parameters
        st.subheader("⚙️ Detection Parameters")
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.01,
            max_value=1.0,
            value=0.25,
            step=0.01
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.01,
            max_value=1.0,
            value=0.7,
            step=0.01
        )
        
        image_size = st.selectbox(
            "Image Size",
            options=[320, 416, 512, 640, 832, 1024],
            index=3
        )
        
        # Model status
        if st.session_state.model_loaded:
            st.success("✅ Model loaded and ready")
        else:
            st.warning("⚠️ Please load a model first")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🖼️ Input Image")
        
        # Image upload
        image_file = st.file_uploader(
            "Upload Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image for object detection"
        )
        
        if image_file is not None:
            try:
                # Load and display image
                image = Image.open(image_file)
                image_np = np.array(image)
                
                # Display image (simple version for compatibility)
                st.image(image, caption="Input Image")
                
                # Image info
                st.info(f"📊 Image size: {image.width} × {image.height} pixels")
                
                # Detection button
                if st.button("🔍 Detect Objects", type="primary"):
                    if st.session_state.model_loaded:
                        with st.spinner("Performing object detection..."):
                            annotated_img, summary_text = perform_detection(
                                image_np, conf_threshold, iou_threshold, image_size
                            )
                            
                            # Store results
                            st.session_state.last_result = {
                                'annotated_img': annotated_img,
                                'summary_text': summary_text
                            }
                    else:
                        st.error("❌ Please load a model first")
                        
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
    
    with col2:
        st.header("🎯 Detection Results")
        
        if hasattr(st.session_state, 'last_result'):
            result = st.session_state.last_result
            
            # Display annotated image
            if result['annotated_img'] is not None:
                st.image(result['annotated_img'], caption="Detection Results")
            
            # Display summary
            st.markdown(result['summary_text'])
        else:
            st.info("🔍 Upload an image and click 'Detect Objects' to see results")
    
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