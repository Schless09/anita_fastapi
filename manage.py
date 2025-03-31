#!/usr/bin/env python
import os
import signal
import subprocess
import sys
import time
import psutil
import logging
import traceback
import socket
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def stream_output(process):
    """Stream output from the process in real-time."""
    def stream_pipe(pipe, prefix=''):
        try:
            with pipe:
                for line in iter(pipe.readline, b''):
                    print(f"{prefix}{line.decode().strip()}")
                    sys.stdout.flush()
        except (IOError, ValueError):
            pass
    
    # Create threads for stdout and stderr
    stdout_thread = threading.Thread(target=stream_pipe, args=(process.stdout,))
    stderr_thread = threading.Thread(target=stream_pipe, args=(process.stderr, 'ERROR: '))
    
    # Start threads
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    return stdout_thread, stderr_thread

def find_server_pid():
    """Find any running uvicorn server processes."""
    pids = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and ('uvicorn' in cmdline[0] or 'python' in cmdline[0]) and 'app.main:app' in ' '.join(cmdline):
                pids.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids

def kill_server(pids):
    """Gracefully kill the server processes."""
    if not isinstance(pids, list):
        pids = [pids]
    
    for pid in pids:
        try:
            logger.info(f"Attempting to gracefully terminate process {pid}")
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            logger.warning(f"Process {pid} already terminated")
        except Exception as e:
            logger.error(f"Error killing process {pid}: {e}")
    
    # Wait and force kill if necessary
    time.sleep(2)
    for pid in pids:
        try:
            if psutil.pid_exists(pid):
                logger.info(f"Force killing process {pid}")
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass

def ensure_port_available(port):
    """Ensure the port is available by killing any processes using it."""
    try:
        # Try to kill any existing uvicorn processes first
        pids = find_server_pid()
        if pids:
            kill_server(pids)
            time.sleep(2)
        
        # Use lsof to find and kill any process using the port
        subprocess.run(
            f"lsof -i :{port} | awk 'NR!=1 {{print $2}}' | xargs kill -9 2>/dev/null || true",
            shell=True
        )
        time.sleep(1)
        
        # Final check with socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            s.close()
            return True
    except socket.error:
        logger.error(f"Port {port} is still in use after cleanup attempts")
        return False
    except Exception as e:
        logger.error(f"Error ensuring port availability: {e}")
        return False

def start_server(port=8000, reload=True):
    """Start the uvicorn server with the specified configuration."""
    try:
        # Ensure the port is available
        if not ensure_port_available(port):
            logger.error(f"Could not free port {port}")
            sys.exit(1)
        
        # Build the command
        cmd = [
            sys.executable,  # Use the current Python interpreter
            "-m",
            "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "info"
        ]
        if reload:
            cmd.append("--reload")
        
        # Start the server
        logger.info(f"Starting server on port {port}...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Ensure we're in the correct directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Start the process with pipe for output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # Unbuffered
            env=os.environ.copy()  # Pass through all environment variables
        )
        
        # Set up output streaming
        stdout_thread, stderr_thread = stream_output(process)
        
        # Wait a moment to check if the process is still running
        time.sleep(2)
        if process.poll() is not None:
            # Process terminated immediately
            logger.error("Server failed to start.")
            sys.exit(1)
        
        logger.info(f"Server started successfully (PID: {process.pid})")
        
        # Keep the main thread running
        try:
            while True:
                time.sleep(1)
                if process.poll() is not None:
                    break
        except KeyboardInterrupt:
            logger.info("\nReceived keyboard interrupt. Shutting down...")
            kill_server(process.pid)
        
        return process
        
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def restart_server(port=8000):
    """Restart the server by killing existing process and starting a new one."""
    pids = find_server_pid()
    if pids:
        logger.info(f"Found existing server processes: {pids}")
        kill_server(pids)
        time.sleep(2)  # Give more time for cleanup
    
    return start_server(port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manage the FastAPI server")
    parser.add_argument("action", choices=["start", "stop", "restart"], help="Action to perform")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    try:
        if args.action == "start":
            start_server(port=args.port, reload=not args.no_reload)
        elif args.action == "stop":
            pids = find_server_pid()
            if pids:
                kill_server(pids)
                logger.info(f"Server stopped (PIDs: {pids})")
            else:
                logger.info("No running server found")
        elif args.action == "restart":
            restart_server(port=args.port)
    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt. Shutting down...")
        pids = find_server_pid()
        if pids:
            kill_server(pids)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1) 