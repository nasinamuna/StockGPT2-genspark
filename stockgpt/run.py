import os
import sys
import subprocess
import argparse
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_setup():
    """Run the setup script to prepare the environment."""
    try:
        logger.info("Running setup script...")
        subprocess.run([sys.executable, "setup.py"], check=True)
        logger.info("Setup completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during setup: {e}")
        sys.exit(1)

def start_backend():
    """Start the backend FastAPI server."""
    try:
        logger.info("Starting backend server...")
        # Fix for absolute imports in app.py
        os.environ['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Start the app inside the src directory
        proc = subprocess.Popen([
            sys.executable, 
            "-m", "uvicorn", 
            "app:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], cwd="src")
        
        return proc
    except Exception as e:
        logger.error(f"Error starting backend: {e}")
        sys.exit(1)

def start_frontend():
    """Start the frontend React development server."""
    # Get the path to the stockgpt-frontend directory (one level up)
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stockgpt-frontend'))
    
    # Check if frontend directory exists
    if not os.path.exists(frontend_dir):
        logger.warning(f"Frontend directory '{frontend_dir}' not found. Skipping frontend startup.")
        logger.info("To create a basic frontend, run: 'npx create-react-app stockgpt-frontend' in the parent directory")
        return None
        
    try:
        logger.info(f"Starting frontend server from {frontend_dir}...")
        
        # Change to the frontend directory
        current_dir = os.getcwd()
        os.chdir(frontend_dir)
        
        # Check if node_modules exists, if not run npm install
        if not os.path.exists("node_modules"):
            logger.info("Installing frontend dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        # Start the frontend development server
        proc = subprocess.Popen(["npm", "start"])
        
        # Return to the original directory
        os.chdir(current_dir)
        
        return proc
    except Exception as e:
        logger.error(f"Error starting frontend: {e}")
        # Make sure we return to the original directory if there's an error
        if os.getcwd() != current_dir:
            os.chdir(current_dir)
        return None

def main():
    """Main entry point for running the StockGPT application."""
    parser = argparse.ArgumentParser(description="StockGPT Runner")
    parser.add_argument("--setup", action="store_true", help="Run setup only")
    parser.add_argument("--backend", action="store_true", help="Run backend only")
    parser.add_argument("--frontend", action="store_true", help="Run frontend only")
    
    args = parser.parse_args()
    
    # Run setup if requested or if running all components
    if args.setup or (not args.backend and not args.frontend):
        run_setup()
    
    processes = []
    
    # Start backend if requested or if running all components
    if args.backend or (not args.setup and not args.frontend):
        backend_proc = start_backend()
        if backend_proc:
            processes.append(backend_proc)
        
    # Start frontend if requested or if running all components
    if args.frontend or (not args.setup and not args.backend):
        frontend_proc = start_frontend()
        if frontend_proc:
            processes.append(frontend_proc)
    
    if processes:
        try:
            logger.info("All components started. Press Ctrl+C to stop.")
            # Keep the script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping all processes...")
            for proc in processes:
                proc.terminate()
            
            logger.info("All processes stopped.")

if __name__ == "__main__":
    main() 