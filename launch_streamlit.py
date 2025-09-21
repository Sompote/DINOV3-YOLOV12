#!/usr/bin/env python3
"""
YOLOv12-DINO Streamlit App Launcher
Configures environment for large file uploads before starting Streamlit
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up environment variables for large file uploads."""
    
    # Set environment variables
    os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '5000'
    os.environ['STREAMLIT_SERVER_MAX_MESSAGE_SIZE'] = '5000'
    
    # Create .streamlit directory
    config_dir = Path('.streamlit')
    config_dir.mkdir(exist_ok=True)
    
    # Create config file
    config_content = """[server]
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

[logger]
level = "info"
"""
    
    config_file = config_dir / 'config.toml'
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("✅ Configuration set for uploads up to 5GB")

def launch_streamlit(use_simple=False):
    """Launch Streamlit with proper configuration."""
    
    app_file = 'simple_streamlit_app.py' if use_simple else 'streamlit_app.py'
    app_name = 'Simple' if use_simple else 'Full'
    
    print(f"🚀 Starting YOLOv12-DINO {app_name} Streamlit App...")
    print("🌐 Access the app at: http://localhost:8501")
    print("📁 Upload size limit: 5GB")
    print("-" * 50)
    
    # Launch Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', app_file,
        '--server.maxUploadSize=5000',
        '--server.maxMessageSize=5000',
        '--browser.gatherUsageStats=false'
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Streamlit app...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        return False
    
    return True

def main():
    """Main launcher function."""
    
    print("🎯 YOLOv12-DINO Streamlit App Launcher")
    print("=" * 50)
    
    # Check which apps are available
    has_full = Path('streamlit_app.py').exists()
    has_simple = Path('simple_streamlit_app.py').exists()
    
    if not has_full and not has_simple:
        print("❌ No Streamlit app files found in current directory")
        print("Please run this script from the YOLOv12 directory")
        return False
    
    # Choose app version
    if has_full and has_simple:
        print("📋 Available apps:")
        print("1. Full app (streamlit_app.py) - Advanced features, charts, tables")
        print("2. Simple app (simple_streamlit_app.py) - Basic functionality, better compatibility")
        
        choice = input("\nChoose app version (1 or 2, default=2): ").strip()
        use_simple = choice != "1"
    else:
        use_simple = has_simple
    
    # Setup environment
    setup_environment()
    
    # Launch app
    return launch_streamlit(use_simple)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)