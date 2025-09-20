# Lexi-Core: AI Legal Navigator 🤖⚖️

Lexi-Core is a cutting-edge AI-powered legal assistant that combines advanced OCR technology with Gemini AI to provide intelligent document analysis and legal consultation services.

## ✨ Features

### 🔍 **Advanced Document Analysis**
- **Multi-Strategy OCR**: Automatically tries multiple OCR approaches and selects the best result
- **AI-Powered Analysis**: Gemini AI integration for intelligent document summarization
- **Smart Quality Detection**: Automatically detects and provides guidance for poor-quality documents
- **Structured Field Extraction**: Identifies and extracts form fields, IDs, and structured data

### 🤖 **AI Legal Assistant**
- **Gemini 1.5 Flash Integration**: Advanced AI for legal question answering
- **Document-Aware Responses**: Context-aware analysis based on uploaded documents
- **Legal Clause Identification**: Automatically identifies important clauses and obligations
- **Risk Assessment**: AI-powered detection of potential legal risks

### 📄 **Document Support**
- **Multiple Formats**: PDF, images (JPG, PNG), and various document types
- **ID Card Processing**: Specialized processing for student IDs, employment forms
- **Employment Documents**: Enhanced parsing for job applications and forms
- **General Legal Documents**: Contracts, agreements, and legal text analysis

### 🎨 **Modern UI/UX**
- **Glass Morphism Design**: Modern, professional interface
- **Thinking Dots Animation**: Elegant loading indicators
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Analysis**: Live document processing with progress indicators

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+
- Tesseract OCR
- Google Gemini API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amitava121/Lexi-Core.git
   cd Lexi-Core
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**
   - **macOS**: `brew install tesseract`
   - **Ubuntu**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Configure API Keys**
   - Edit `script.js` and add your Gemini API key:
   ```javascript
   const GEMINI_API_KEY = 'your-gemini-api-key-here';
   ```

5. **Start the system**
   ```bash
   python3 system_manager.py
   ```

6. **Access the application**
   - Open your browser to `http://localhost:3000`

## 🔧 **System Architecture**

### Backend Services
- **Enhanced Tesseract OCR Server** (`enhanced_tesseract_ocr.py`): Port 5001
- **Web Server** (Python HTTP): Port 3000
- **System Manager** (`system_manager.py`): Orchestrates all services

### Frontend
- **Modern Web Interface**: HTML5, CSS3, JavaScript ES6+
- **AI Integration**: Direct Gemini API calls
- **Real-time OCR**: Asynchronous document processing

### OCR Processing Pipeline
1. **Multi-Strategy Approach**: Tries accurate, balanced, and fast modes
2. **Quality Scoring**: Combines confidence and readability metrics
3. **Best Result Selection**: Automatically chooses optimal OCR result
4. **Structured Analysis**: Extracts fields, clauses, and key information

## 🛠 **Usage**

### Document Upload & Analysis
1. Click the attachment button or drag & drop a document
2. System automatically processes with multiple OCR strategies
3. AI analyzes content for legal significance
4. Receive comprehensive analysis with:
   - Document classification
   - AI-generated summary
   - Extracted form fields
   - Important clauses
   - Obligations and risks

### Legal Consultation
- Ask legal questions in natural language
- Get AI-powered responses with legal context
- Document-aware answers when files are uploaded
- Access to legal case law references

### Quality Assurance
- Automatic quality detection for uploaded images
- Recommendations for improving OCR results
- Multiple OCR mode comparison
- Confidence scoring and readability analysis

## 📁 **Project Structure**

```
legal_navigator/
├── index.html              # Main web interface
├── script.js              # Frontend JavaScript logic
├── style.css              # UI styling
├── enhanced_tesseract_ocr.py # Advanced OCR server
├── system_manager.py       # System orchestration
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
├── README.md              # This file
├── PYTHON_SCRIPTS.md      # Python scripts documentation
└── .gitignore            # Git ignore rules
```

## ⚙️ **Configuration**

### OCR Settings
- **Modes Available**: fast, balanced, accurate
- **Multi-server Support**: Primary + backup OCR servers
- **Quality Thresholds**: Configurable confidence and readability limits

### AI Integration
- **Gemini API**: Configure in `script.js`
- **Court Listener API**: Legal case law integration
- **Response Customization**: Modify AI prompts for specific needs

## 🔒 **Security**

- **API Key Management**: Keep API keys secure and use environment variables
- **CORS Configuration**: Properly configured for local development
- **File Upload Safety**: Validates file types and sizes
- **No Data Storage**: Documents processed in memory only

## 🧪 **Testing**

- **OCR Testing**: `test_ocr.py`, `test_ocr_detailed.py`
- **System Testing**: `test_web_workflow.py`
- **Health Checks**: Built-in endpoint monitoring

## 📊 **Performance**

- **OCR Processing**: 2-5 seconds per document
- **AI Analysis**: 1-3 seconds for responses
- **Multi-strategy**: Automatically optimized for quality
- **Memory Efficient**: No persistent storage required

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Tesseract OCR** for text extraction capabilities
- **Google Gemini** for AI integration
- **Court Listener** for legal database access
- **Flask & CORS** for backend infrastructure

## 🆘 **Support**

If you encounter any issues:
1. Check the system health at `http://localhost:5001/health`
2. Review logs in the terminal
3. Ensure all dependencies are installed
4. Verify API keys are configured correctly

## 🔄 **Updates**

### Recent Improvements
- **Multi-Strategy OCR**: Enhanced accuracy with multiple processing modes
- **AI-Powered Analysis**: Full Gemini integration for document understanding
- **Quality Detection**: Smart handling of poor-quality images
- **Enhanced UI**: Modern glass morphism design with thinking animations
- **Better Error Handling**: Comprehensive error messages and recovery

---

**Built with ❤️ for legal professionals and document analysis**
