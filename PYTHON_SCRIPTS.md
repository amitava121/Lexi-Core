# Legal Navigator System Manager

## Overview
A unified Python script that provides comprehensive system management for the Legal Navigator application with cross-platform compatibility and robust error handling.

## Unified Script

### 🚀 `system_manager.py` - Complete System Manager
**Single script for all Legal Navigator system operations**

#### Features:
- **Unified Management** - Start, stop, restart, status, and clear ports in one script
- **Cross-platform compatibility** (Windows, macOS, Linux)
- **Intelligent port management** - automatically frees busy ports
- **Health monitoring** - validates all services are running properly
- **Process monitoring** - restarts failed services automatically
- **Browser integration** - opens application automatically
- **Graceful shutdown** - handles Ctrl+C and system signals
- **Comprehensive status reporting** - shows all system information
- **Port clearing** - cleans up reserved ports completely

#### Usage:
```bash
# Start the system (default action)
./system_manager.py
./system_manager.py start

# Stop the system
./system_manager.py stop

# Restart the system
./system_manager.py restart

# Check system status
./system_manager.py status

# Clear reserved ports
./system_manager.py clear

# Alternative Python execution
python3 system_manager.py [action]
```

#### Available Actions:

##### 🚀 **START** (default)
- **System Validation**: Checks Python version and required files
- **Port Cleanup**: Automatically frees ports 5001 and 3000 if occupied
- **OCR Server**: Starts `simple_ocr_server.py` on port 5001
- **Web Server**: Starts Python HTTP server on port 3000
- **Health Checks**: Validates both servers are responding
- **Browser Launch**: Opens http://localhost:3000 automatically
- **Monitoring**: Continuously monitors processes for failures
- **Status Display**: Shows comprehensive system status

##### 🛑 **STOP**
- **Graceful Termination**: Sends SIGTERM first, then SIGKILL if needed
- **Port Cleanup**: Kills processes using ports 5001 and 3000
- **Process Search**: Finds and stops Python HTTP servers and OCR servers
- **Comprehensive Cleanup**: Ensures no orphaned processes remain
- **Verification**: Confirms all processes are stopped

##### 🔄 **RESTART**
- **Clean Stop**: Gracefully stops all running services
- **Port Cleanup**: Ensures all ports are freed
- **Fresh Start**: Starts system with clean state
- **Full Validation**: Checks all components after restart

##### 📊 **STATUS**
- **Port Analysis**: Checks which ports are in use
- **Health Checks**: Tests OCR and Web server endpoints
- **Service Status**: Shows if services are healthy or failed
- **Comprehensive Report**: Displays complete system state

##### 🧹 **CLEAR**
- **Port Liberation**: Frees ports 5001 and 3000 completely
- **Process Cleanup**: Kills related Python processes
- **Reserved Port Clearing**: Ensures no lingering connections
- **Clean State**: Prepares system for fresh start

## System Requirements

### Python Version:
- **Required**: Python 3.6 or higher
- **Recommended**: Python 3.8+

### Required Files:
- `simple_ocr_server.py` - OCR processing server
- `index.html` - Main application interface
- `script.js` - Frontend logic with optimizations
- `style.css` - Styling with compact design

### Python Libraries:
The script uses only standard library modules:
- `os`, `sys`, `time`, `signal` - System operations
- `subprocess`, `threading` - Process management
- `pathlib` - Cross-platform path handling
- `webbrowser` - Browser integration
- `requests`, `json` - HTTP communication
- `socket` - Network utilities
- `argparse` - Command-line argument parsing

## System Architecture

### Port Usage:
- **Port 5001**: OCR Server (Flask-based)
  - Health endpoint: `http://localhost:5001/health`
  - OCR processing: `http://localhost:5001/ocr`
- **Port 3000**: Web Server (Python HTTP server)
  - Application: `http://localhost:3000`
  - Static files: HTML, CSS, JavaScript

### Process Management:
- **Main Process**: Python launcher script
- **Child Process 1**: OCR server (`simple_ocr_server.py`)
- **Child Process 2**: Web server (`python -m http.server 3000`)
- **Monitor Thread**: Background process health monitoring

## Troubleshooting

### Common Issues:

#### Port Already in Use:
```
❌ Port 5001 is busy, attempting to free it...
```
**Solution**: The script automatically handles this with the clear action:
```bash
# Clear all ports manually
./system_manager.py clear

# Check what's using the port (manual)
lsof -i :5001
lsof -i :3000
```

#### Permission Errors:
```bash
# Make script executable
chmod +x system_manager.py

# Run with explicit Python
python3 system_manager.py start
```

#### Missing Dependencies:
```
❌ Error: Missing required files: simple_ocr_server.py
```
**Solution**: Run the script from the correct directory containing all Legal Navigator files.

#### OCR Server Won't Start:
```
❌ OCR Server failed to start properly
```
**Possible causes**:
- Missing OCR dependencies (Tesseract)
- Python path issues  
- File permissions

**Debug steps**:
```bash
# Check system status
./system_manager.py status

# Test OCR server manually
python3 simple_ocr_server.py

# Check Tesseract installation
tesseract --version
```

### Manual Process Management:

#### Check Running Processes:
```bash
# Check system status with built-in command
./system_manager.py status

# Manual process checking
ps aux | grep -E "(simple_ocr_server|http.server)"

# Check specific ports
netstat -an | grep -E "(5001|3000)"
```

#### Manual Cleanup:
```bash
# Use built-in clear command
./system_manager.py clear

# Manual cleanup if needed
pkill -f "python.*http.server"
pkill -f "simple_ocr_server"

# Force kill by port
lsof -ti:5001 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

## Performance Features

### Speed Optimizations:
- **No artificial delays** - removed 1.1 second response delays
- **Immediate file processing** - OCR analysis starts instantly
- **Concurrent health checks** - parallel validation of services
- **Efficient monitoring** - lightweight background thread

### UI Improvements:
- **Compact thinking dots** - 60% smaller animation (5px dots, 8px padding)
- **Markdown formatting** - proper bold, italic, heading rendering
- **Clean branding** - professional Lexi-Core AI presentation

### System Benefits:
- **Unified Management** - single script handles all operations
- **One-command operations** - start, stop, status, clear with simple commands
- **Automatic browser opening** - immediate access to application
- **Health monitoring** - automatic restart of failed services
- **Graceful shutdown** - clean termination on Ctrl+C
- **Port management** - automatic clearing of reserved ports
- **Status checking** - comprehensive system health reports

## Integration with Legal Navigator

### Core Features Enabled:
- **🤖 Lexi-Core AI**: Gemini 1.5 Flash integration for legal analysis
- **📄 Document OCR**: Enhanced Tesseract processing with multiple PSM modes
- **⚖️ Legal Research**: Court Listener API integration
- **⚡ Fast Responses**: Sub-second response times for simple queries
- **🎨 Professional UI**: Clean interface with proper formatting

### API Endpoints Ready:
- **AI Chat**: Frontend handles Gemini API communication
- **OCR Processing**: Upload and analyze legal documents
- **Court Search**: Integration with legal case databases
- **Health Monitoring**: System status and diagnostics

## Security Considerations

### Process Isolation:
- Scripts run with user permissions
- No root/admin privileges required
- Process cleanup prevents orphaned services

### Port Security:
- Services bind to localhost only
- No external network exposure
- Standard HTTP ports for development

### Error Handling:
- Graceful failure recovery
- Comprehensive error logging
- Safe shutdown procedures

---

**Created**: September 18, 2025  
**Version**: 2.0 (Unified Manager)  
**Compatibility**: Python 3.6+, Windows/macOS/Linux  
**Part of**: Legal Navigator System v2025091812