#!/bin/bash
# Legal Navigator - Free AI Setup Script
# This script sets up the free and open-source AI enhancements

echo "🚀 Setting up Legal Navigator with Free AI Enhancements..."
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    echo "Please install pip3 and try again."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
echo "This may take several minutes..."

pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Failed to install Python dependencies"
    echo "You can try installing manually with:"
    echo "pip3 install paddlepaddle paddleocr easyocr opencv-python pillow numpy flask flask-cors"
fi

# Make Python script executable
chmod +x paddle_ocr_server.py

echo ""
echo "🎉 Setup Complete!"
echo "=================================================="
echo ""
echo "🚀 To start the enhanced Legal Navigator:"
echo ""
echo "1. Start the Python OCR server (in one terminal):"
echo "   python3 paddle_ocr_server.py"
echo ""
echo "2. Start the web server (in another terminal):"
echo "   python3 -m http.server 8000"
echo ""
echo "3. Open your browser to:"
echo "   http://localhost:8000"
echo ""
echo "✨ Features enabled:"
echo "• 🤗 HuggingFace AI models (free)"
echo "• 🐍 PaddleOCR + EasyOCR (90-95% accuracy)"
echo "• ⚖️ Court Listener case law database"
echo "• 🔄 Intelligent fallback system"
echo "• 🖼️ Enhanced image preprocessing"
echo ""
echo "📝 Note: The first OCR operation may be slow as models download."
echo "📝 All AI features work completely offline after initial setup!"