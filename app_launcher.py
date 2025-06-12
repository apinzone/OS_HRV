import subprocess
import sys
import os

def launch_app():
    # Get the directory where the executable is located
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        app_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to your main streamlit file
    app_path = os.path.join(app_dir, "enhanced_professional_gui.py")
    
    # Launch streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, "--server.headless", "true"])

if __name__ == "__main__":
    launch_app()