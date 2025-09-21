#!/bin/bash

# YOLOv12-DINO Streamlit App Launcher with Large File Support
# This script configures Streamlit to handle large model files

echo "🚀 Starting YOLOv12-DINO Streamlit App with Large File Support..."

# Set environment variables for large file uploads
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5000
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=5000

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Ensure config file exists with correct settings
cat > .streamlit/config.toml << EOF
[server]
maxUploadSize = 5000
maxMessageSize = 5000
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1e88e5"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
EOF

echo "✅ Configuration set for uploads up to 5GB"
echo "🌐 Starting Streamlit server..."

# Launch Streamlit with explicit config
streamlit run streamlit_app.py \
    --server.maxUploadSize=5000 \
    --server.maxMessageSize=5000 \
    --browser.gatherUsageStats=false