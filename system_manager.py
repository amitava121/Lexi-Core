#!/usr/bin/env python3
"""
Legal Navigator System Manager
A unified Python script to start, stop, and manage the Legal Navigator system
"""

import os
import sys
import time
import signal
import subprocess
import threading
import webbrowser
import argparse
from pathlib import Path
import requests
import json

class LegalNavigatorManager:
    def __init__(self):
        self.ocr_process = None
        self.web_process = None
        self.running = False
        self.base_dir = Path(__file__).parent.absolute()
        self.monitor_thread = None
        
    def print_banner(self, action=""):
        """Print banner"""
        print(f"🚀 Legal Navigator System Manager{' - ' + action if action else ''}")
        print("=" * 50)
        print("🤖 Lexi-Core AI Legal Assistant")
        print("📄 Document OCR & Analysis")
        print("⚖️ Legal Document Processing")
        print("=" * 50)
        print()
    
    def check_requirements(self):
        """Check system requirements"""
        print("🔍 Checking system requirements...")
        
        # Check if we're in the right directory
        required_files = ['enhanced_tesseract_ocr.py', 'index.html', 'script.js']
        missing_files = []
        
        for file in required_files:
            if not (self.base_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Error: Missing required files: {', '.join(missing_files)}")
            print(f"   Please run this script from the legal_navigator directory")
            print(f"   Current directory: {self.base_dir}")
            return False
        
        # Check Python version
        if sys.version_info < (3, 6):
            print(f"❌ Error: Python 3.6+ required, found {sys.version}")
            return False
        
        print("✅ All requirements satisfied")
        return True
    
    def is_port_in_use(self, port):
        """Check if a port is in use"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        except:
            return False
    
    def clear_port(self, port, service_name="Service"):
        """Clear processes using a specific port"""
        print(f"🔍 Checking {service_name} (port {port})...")
        
        if not self.is_port_in_use(port):
            print(f"   ✅ Port {port} is already free")
            return True
        
        print(f"   🧹 Clearing port {port}...")
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(['netstat', '-ano'], 
                                      capture_output=True, text=True)
                lines = result.stdout.split('\n')
                for line in lines:
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            try:
                                subprocess.run(['taskkill', '/F', '/PID', pid], 
                                             capture_output=True)
                                print(f"   🛑 Killed process {pid}")
                            except:
                                pass
            else:  # Unix-like
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        try:
                            pid_int = int(pid)
                            os.kill(pid_int, signal.SIGTERM)
                            print(f"   🛑 Sent SIGTERM to PID {pid}")
                        except:
                            try:
                                os.kill(pid_int, signal.SIGKILL)
                                print(f"   💀 Force killed PID {pid}")
                            except:
                                pass
                    
                    # Wait for graceful shutdown
                    time.sleep(2)
                    
                    # Force kill any remaining
                    result2 = subprocess.run(['lsof', '-ti', f':{port}'], 
                                           capture_output=True, text=True)
                    if result2.returncode == 0 and result2.stdout.strip():
                        pids = result2.stdout.strip().split('\n')
                        for pid in pids:
                            try:
                                os.kill(int(pid), signal.SIGKILL)
                                print(f"   💀 Force killed remaining PID {pid}")
                            except:
                                pass
            
            # Final verification
            time.sleep(1)
            if not self.is_port_in_use(port):
                print(f"   ✅ Port {port} cleared successfully")
                return True
            else:
                print(f"   ⚠️ Port {port} may still be occupied")
                return False
                
        except Exception as e:
            print(f"   ❌ Error clearing port {port}: {e}")
            return False
    
    def clear_all_ports(self):
        """Clear all ports used by the system"""
        print("🧹 Clearing all reserved ports...")
        ocr_cleared = self.clear_port(5001, "OCR Server")
        web_cleared = self.clear_port(3000, "Web Server")
        
        # Also clear any related Python processes
        self.clear_python_processes()
        
        if ocr_cleared and web_cleared:
            print("✅ All ports cleared successfully!")
            return True
        else:
            print("⚠️ Some ports may still be occupied")
            return False
    
    def clear_python_processes(self):
        """Clear related Python processes"""
        try:
            if os.name != 'nt':  # Unix-like systems
                # Kill Python HTTP servers
                result = subprocess.run(['pgrep', '-f', 'python.*http.server'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            print(f"   🛑 Stopped Python HTTP server (PID {pid})")
                        except:
                            pass
                
                # Kill OCR servers
                result = subprocess.run(['pgrep', '-f', 'ultra_ocr_server'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            print(f"   🛑 Stopped OCR server (PID {pid})")
                        except:
                            pass
        except:
            pass
    
    def start_ocr_server(self):
        """Start the OCR server"""
        print("🔧 Starting OCR Server (port 5001)...")
        
        # Clear port first
        self.clear_port(5001, "OCR Server")
        
        try:
            # Start Enhanced Tesseract OCR server
            print("🧠 Starting Enhanced Tesseract OCR server...")
            self.ocr_process = subprocess.Popen(
                [sys.executable, 'enhanced_tesseract_ocr.py'],
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            for attempt in range(10):
                time.sleep(1)
                try:
                    response = requests.get('http://localhost:5001/health', timeout=2)
                    if response.status_code == 200:
                        data = response.json()
                        print(f"✅ OCR Server started (PID: {self.ocr_process.pid})")
                        print(f"   Status: {data.get('message', 'Running')}")
                        return True
                except:
                    continue
            
            print("❌ OCR Server failed to start properly")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start OCR Server: {e}")
            return False
    
    def start_web_server(self):
        """Start the web server"""
        print("🌐 Starting Web Server (port 3000)...")
        
        # Clear port first
        self.clear_port(3000, "Web Server")
        
        try:
            # Start web server
            self.web_process = subprocess.Popen(
                [sys.executable, '-m', 'http.server', '3000'],
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            for attempt in range(10):
                time.sleep(1)
                try:
                    response = requests.get('http://localhost:3000', timeout=2)
                    if response.status_code == 200:
                        print(f"✅ Web Server started (PID: {self.web_process.pid})")
                        return True
                except:
                    continue
            
            print("❌ Web Server failed to start properly")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start Web Server: {e}")
            return False
    
    def open_browser(self):
        """Open the application in the default browser"""
        print("🌐 Opening Legal Navigator in browser...")
        try:
            webbrowser.open('http://localhost:3000')
            print("   Browser opened successfully")
        except:
            print("   Please manually open: http://localhost:3000")
    
    def monitor_processes(self):
        """Monitor the running processes"""
        while self.running:
            time.sleep(5)
            
            # Check OCR server
            if self.ocr_process and self.ocr_process.poll() is not None:
                print("❌ OCR Server stopped unexpectedly!")
                self.stop_system()
                break
            
            # Check Web server
            if self.web_process and self.web_process.poll() is not None:
                print("❌ Web Server stopped unexpectedly!")
                self.stop_system()
                break
    
    def print_status(self):
        """Print system status"""
        print()
        print("🎉 Legal Navigator System is ready!")
        print("=" * 50)
        print("📍 Application URL: http://localhost:3000")
        print("🔧 OCR Server API: http://localhost:5001")
        print("📊 System Status:")
        if self.ocr_process:
            print(f"   • OCR Server: Running (PID: {self.ocr_process.pid})")
        if self.web_process:
            print(f"   • Web Server: Running (PID: {self.web_process.pid})")
        print()
        print("💡 Features Ready:")
        print("   • Lexi-Core AI Assistant")
        print("   • Document OCR Analysis") 
        print("   • Legal Document Processing")
        print("   • Fast Response Times")
        print("   • Enhanced Markdown Formatting")
        print()
        print("Commands:")
        print("   • Ctrl+C to stop the system")
        print(f"   • python3 {sys.argv[0]} stop")
        print(f"   • python3 {sys.argv[0]} clear")
        print("=" * 50)
        print()
    
    def start_system(self):
        """Start the Legal Navigator system"""
        self.print_banner("START")
        
        # Check requirements
        if not self.check_requirements():
            return False
        
        # Clear ports first
        print("🧹 Preparing system (clearing ports)...")
        self.clear_all_ports()
        
        # Start OCR server
        if not self.start_ocr_server():
            return False
        
        # Start web server
        if not self.start_web_server():
            self.stop_system()
            return False
        
        # Open browser
        self.open_browser()
        
        # Print status
        self.print_status()
        
        # Set running flag
        self.running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        self.monitor_thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        return True
    
    def stop_system(self):
        """Stop the Legal Navigator system"""
        self.print_banner("STOP")
        self.running = False
        
        print("🛑 Shutting down Legal Navigator System...")
        
        if self.ocr_process:
            print("   Stopping OCR Server...")
            try:
                self.ocr_process.terminate()
                self.ocr_process.wait(timeout=5)
            except:
                try:
                    self.ocr_process.kill()
                except:
                    pass
            self.ocr_process = None
        
        if self.web_process:
            print("   Stopping Web Server...")
            try:
                self.web_process.terminate()
                self.web_process.wait(timeout=5)
            except:
                try:
                    self.web_process.kill()
                except:
                    pass
            self.web_process = None
        
        # Clear all ports to ensure clean shutdown
        self.clear_all_ports()
        
        print("✅ Legal Navigator System stopped successfully!")
        print("=" * 50)
    
    def check_status(self):
        """Check system status"""
        self.print_banner("STATUS")
        
        print("🔍 Checking system status...")
        print()
        
        # Check ports
        ocr_running = self.is_port_in_use(5001)
        web_running = self.is_port_in_use(3000)
        
        print("📊 Port Status:")
        print(f"   • Port 5001 (OCR Server): {'🟢 In Use' if ocr_running else '🔴 Free'}")
        print(f"   • Port 3000 (Web Server): {'🟢 In Use' if web_running else '🔴 Free'}")
        print()
        
        # Check endpoints
        if ocr_running:
            try:
                response = requests.get('http://localhost:5001/health', timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    print("🔧 OCR Server: ✅ Healthy")
                    print(f"   Message: {data.get('message', 'Unknown')}")
                    engines = data.get('available_engines', [])
                    print(f"   Engines: {', '.join(engines)}")
                else:
                    print("🔧 OCR Server: ⚠️ Responding but unhealthy")
            except:
                print("🔧 OCR Server: ❌ Not responding")
        
        if web_running:
            try:
                response = requests.get('http://localhost:3000', timeout=2)
                if response.status_code == 200:
                    print("🌐 Web Server: ✅ Healthy")
                    print("   URL: http://localhost:3000")
                else:
                    print("🌐 Web Server: ⚠️ Responding but unhealthy")
            except:
                print("🌐 Web Server: ❌ Not responding")
        
        print()
        if ocr_running and web_running:
            print("🎉 Legal Navigator System is running!")
        elif ocr_running or web_running:
            print("⚠️ Legal Navigator System is partially running")
        else:
            print("💤 Legal Navigator System is stopped")
        
        print("=" * 50)
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C and other signals"""
        self.stop_system()
        sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Legal Navigator System Manager')
    parser.add_argument('action', nargs='?', default='start',
                       choices=['start', 'stop', 'restart', 'status', 'clear'],
                       help='Action to perform (default: start)')
    
    args = parser.parse_args()
    manager = LegalNavigatorManager()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)
    
    try:
        if args.action == 'start':
            success = manager.start_system()
        elif args.action == 'stop':
            manager.stop_system()
            success = True
        elif args.action == 'restart':
            manager.stop_system()
            time.sleep(2)
            success = manager.start_system()
        elif args.action == 'status':
            manager.check_status()
            success = True
        elif args.action == 'clear':
            manager.print_banner("CLEAR PORTS")
            success = manager.clear_all_ports()
        else:
            print("❌ Unknown action")
            success = False
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()