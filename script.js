// Legal Navigator - Fresh Clean Implementation
console.log('🚀 Legal Navigator Starting Fresh...');

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing...');
    
    // Get essential elements
    const chatWindow = document.getElementById('chat-window');
    const historyList = document.getElementById('history-list');
    const newChatBtn = document.getElementById('new-chat-button');
    const toggleHistoryBtn = document.getElementById('toggle-history-btn');
    const historyContent = document.querySelector('.history-content');
    const textInput = document.getElementById('text-input');
    const sendButton = document.getElementById('send-button');
    const attachFileBtn = document.getElementById('attach-file-button');
    const fileInput = document.getElementById('file-input');
    const attachedFileName = document.getElementById('attached-file-name');
    
    console.log('Elements found:', {
        chatWindow: !!chatWindow,
        historyList: !!historyList, 
        newChatBtn: !!newChatBtn,
        toggleHistoryBtn: !!toggleHistoryBtn,
        historyContent: !!historyContent,
        textInput: !!textInput,
        sendButton: !!sendButton,
        attachFileBtn: !!attachFileBtn,
        fileInput: !!fileInput
    });
    
    if (!chatWindow || !historyList) {
        console.error('❌ Critical elements missing');
        return;
    }
    
    // Clear any existing content first
    chatWindow.innerHTML = '';
    historyList.innerHTML = '';
    
    // Global state
    let chatHistory = [];
    let currentChatId = null;
    let attachedFile = null;
    const WELCOME_TEXT = "Hello! I'm Lexi-Core. You can ask a question or upload a legal document (PDF/Image) for analysis.";
    
    // Google Gemini API configuration (primary)
    const GEMINI_API_KEY = 'Replace with your actual API key'; // Replace with your actual API key
    const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent';
    
    // Court Listener API (free legal database)
    const COURT_LISTENER_API = 'https://www.courtlistener.com/api/rest/v3/';
    
    // Legal responses (fallback for API failures)
    const responses = {
        contract: 'A contract is a legally binding agreement between parties. Key elements: offer, acceptance, consideration, legal capacity, and legal purpose.',
        lawsuit: 'To file a lawsuit: determine jurisdiction, file complaint, serve defendant, await response, proceed with discovery.',
        copyright: 'Copyright protects original works automatically. Duration is typically author\'s lifetime plus 70 years.',
        default: 'I\'m Lexi-Core, your AI legal assistant. I can help with contracts, IP law, employment law, and more.'
    };
    
    // File upload functions
    function handleFileAttachment() {
        if (fileInput) {
            fileInput.click();
        }
    }
    
    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            attachedFile = file;
            console.log('📎 File attached:', file.name);
            
            // Update UI to show file is attached
            if (textInput) {
                textInput.placeholder = `File attached: ${file.name} - Add a message or send`;
            }
            
            // Show file name if element exists
            if (attachedFileName) {
                attachedFileName.textContent = file.name;
                attachedFileName.style.display = 'block';
            }
        }
    }
    
    function clearAttachedFile() {
        attachedFile = null;
        if (fileInput) {
            fileInput.value = '';
        }
        if (textInput) {
            textInput.placeholder = 'Ask a question or attach a file...';
        }
        if (attachedFileName) {
            attachedFileName.textContent = '';
            attachedFileName.style.display = 'none';
        }
        console.log('📎 File attachment cleared');
    }
    
    // Extract actual text content from uploaded files
    async function extractDocumentText(file) {
        console.log('📖 Extracting text from:', file.name);
        
        try {
            if (file.type.startsWith('image/')) {
                // Use OCR for images
                return await extractTextFromImage(file);
            } else if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
                // Use PDF.js for PDFs
                return await extractTextFromPDF(file);
            } else {
                return { 
                    success: false, 
                    error: 'Unsupported file type for text extraction',
                    extractedText: '',
                    confidence: 0 
                };
            }
        } catch (error) {
            console.error('❌ Text extraction error:', error);
            return { 
                success: false, 
                error: error.message,
                extractedText: '',
                confidence: 0 
            };
        }
    }

    // Enhanced OCR configuration - Multi-server approach
    const OCR_CONFIG = {
        server_url: 'http://127.0.0.1:5001',     // Enhanced Tesseract OCR server
        backup_servers: [
            'http://127.0.0.1:5002',             // Dual engine OCR (if available)
            'http://127.0.0.1:5003'              // Smart OCR (if available)
        ],
        tesseract_fallback: true,
        max_retries: 3
    };

    // Check if Enhanced Tesseract OCR server is available
    async function checkOCRServer() {
        try {
            const controller = new AbortController();
            const to = setTimeout(() => controller.abort(), 3000);
            const response = await fetch(`${OCR_CONFIG.server_url}/health`, {
                method: 'GET',
                signal: controller.signal
            });
            clearTimeout(to);
            if (response.ok) {
                console.log('✅ Enhanced Tesseract OCR server available');
                return true;
            }
        } catch (error) {
            console.log('❌ Enhanced Tesseract OCR server unavailable:', error.message || error);
        }
        return false;
    }

    // Clean text extraction with Enhanced Tesseract
    async function extractTextFromImage(file) {
        console.log('🔍 Starting Enhanced Tesseract OCR processing...');
        
        // Check if server is available
        const serverAvailable = await checkOCRServer();
        
        if (!serverAvailable) {
            console.log('❌ OCR server not available, falling back to local Tesseract');
            const fallback = await extractTextFromImageTesseract(file);
            // Ensure unified shape
            return fallback && fallback.success ? fallback : {
                success: false,
                extractedText: '',
                confidence: 0,
                method: 'Tesseract.js (Failed) ',
                error: 'Local Tesseract fallback failed'
            };
        }
        
        // Try Enhanced Tesseract OCR with multi-strategy approach
        const ocrResult = await tryTesseractOCR(file);
        if (ocrResult && ocrResult.success) {
            const method = `${ocrResult.engine || 'Enhanced Tesseract'} (${ocrResult.mode || 'auto'})`;
            console.log(`✅ Multi-strategy OCR successful - ${method}: ${ocrResult.confidence}% confidence, ${Math.round((ocrResult.readableRatio || 0) * 100)}% readable`);
            const cleanedText = cleanExtractedText(ocrResult.text);
            return {
                success: true,
                extractedText: cleanedText,
                confidence: ocrResult.confidence,
                method,
                structuredFields: ocrResult.structuredFields || null,
                textLength: ocrResult.text_length || cleanedText.length,
                qualityScore: Math.round(ocrResult.qualityScore || 0),
                readableRatio: Math.round((ocrResult.readableRatio || 0) * 100),
                error: null
            };
        }
        
        // Try fallback to basic Tesseract if server failed  
        console.log('⚠️ OCR server failed, falling back to basic Tesseract...');
        return await extractTextFromImageTesseract(file);
    }

    // Python OCR server integration with multi-strategy approach
    async function tryTesseractOCR(file) {
        try {
            const url = OCR_CONFIG.server_url;
            const base64Data = await fileToBase64(file);
            console.log(`🐍 Trying multi-strategy OCR approach: ${url}`);
            
            // Try multiple OCR modes and pick the best result
            const strategies = ['accurate', 'balanced', 'fast'];
            const results = [];
            
            for (const mode of strategies) {
                try {
                    console.log(`� Trying OCR mode: ${mode}`);
                    const controller = new AbortController();
                    const timeout = setTimeout(() => controller.abort(), 25000); // 25s per strategy
                    
                    const response = await fetch(`${url}/extract-text`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: base64Data, mode: mode }),
                        signal: controller.signal
                    });
                    
                    clearTimeout(timeout);
                    
                    if (response.ok) {
                        const result = await response.json();
                        if (result.success && result.text) {
                            // Calculate quality score based on confidence and text readability
                            const readableChars = result.text.replace(/[^a-zA-Z0-9\s.,;:!?'"()-]/g, '').length;
                            const readableRatio = readableChars / result.text.length;
                            const qualityScore = (result.confidence || 0) * 0.7 + readableRatio * 30;
                            
                            results.push({
                                ...result,
                                mode: mode,
                                readableRatio: readableRatio,
                                qualityScore: qualityScore
                            });
                            
                            console.log(`✅ Mode ${mode}: conf=${result.confidence}%, readable=${Math.round(readableRatio*100)}%, score=${Math.round(qualityScore)}`);
                            
                            // Early exit if we get a very good result
                            if (qualityScore > 80) {
                                console.log(`🎯 Excellent result found with ${mode}, stopping early`);
                                break;
                            }
                        }
                    }
                } catch (error) {
                    console.warn(`⚠️ Mode ${mode} failed:`, error.message);
                    continue;
                }
            }
            
            if (results.length === 0) {
                throw new Error('All OCR strategies failed');
            }
            
            // Pick the best result based on quality score
            const bestResult = results.reduce((best, current) => 
                current.qualityScore > best.qualityScore ? current : best
            );
            
            console.log(`🏆 Best OCR result: ${bestResult.mode} mode with ${Math.round(bestResult.qualityScore)} quality score`);
            
            return bestResult;
            
        } catch (error) {
            console.error('❌ Multi-strategy OCR error:', error.message || error);
            return null;
        }
    }
    
    // Try a specific OCR server with given parameters
    async function tryOCRServer(serverUrl, base64Data, mode) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 20000);
        
        try {
            const response = await fetch(`${serverUrl}/extract-text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Data, mode: mode }),
                signal: controller.signal
            });
            
            clearTimeout(timeout);
            
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            clearTimeout(timeout);
            throw error;
        }
    }
    
    // Calculate quality score for OCR result
    function calculateQualityScore(result) {
        if (!result || !result.text) return 0;
        
        const readableChars = result.text.replace(/[^a-zA-Z0-9\s.,;:!?'"()-]/g, '').length;
        const readableRatio = result.text.length > 0 ? readableChars / result.text.length : 0;
        const confidence = result.confidence || 0;
        
        // Weighted score: 70% confidence, 30% readability
        const qualityScore = confidence * 0.7 + readableRatio * 30;
        
        // Store readability ratio for reporting
        result.readableRatio = readableRatio;
        
        return qualityScore;
    }

    // Convert file to base64
    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
            reader.readAsDataURL(file);
        });
    }

    // Clean and improve extracted text quality
    function cleanExtractedText(text) {
        if (!text || typeof text !== 'string') return '';
        
        // Normalize line breaks and collapse excessive spaces, but keep lines
        let cleanedText = text.replace(/\r\n|\r/g, '\n');
        const lines = cleanedText.split('\n').map(l => l.replace(/\s+/g, ' ').trim());
        const cleanLines = lines.filter(line => {
            if (line.trim().length < 3) return false;
            const readableChars = line.match(/[a-zA-Z0-9\s]/g) || [];
            const readableRatio = readableChars.length / line.length;
            return readableRatio > 0.2;
        });
        cleanedText = cleanLines.join('\n');

        // Avoid aggressive char substitutions that can harm IDs/phones
        // Only light punctuation cleanup
        cleanedText = cleanedText.replace(/[\uFFFD]/g, '').trim();

        // If text is mostly garbage, return a warning
        if (cleanedText.length < 10 || cleanedText.match(/[a-zA-Z]/g)?.length < 5) {
            return 'WARNING: The image quality may be too poor for accurate text extraction. Please try uploading a clearer image or a different file format.';
        }
        
        return cleanedText.trim();
    }

    // Original Tesseract OCR (fallback)
    async function extractTextFromImageTesseract(file) {
        try {
            console.log('🔍 Starting Tesseract OCR for image...');
            
            const { data } = await Tesseract.recognize(file, 'eng', {
                logger: m => {
                    if (m.status === 'recognizing text') {
                        console.log(`Tesseract Progress: ${Math.round(m.progress * 100)}%`);
                    }
                }
            });
            
            console.log('✅ Tesseract OCR completed successfully');
            console.log('📄 Extracted text length:', data.text.length);
            
            // Clean the extracted text
            const cleanedText = cleanExtractedText(data.text);
            
            return {
                success: true,
                extractedText: cleanedText,
                confidence: data.confidence,
                method: 'Tesseract.js',
                error: null
            };
            
        } catch (error) {
            console.error('❌ Tesseract OCR Error:', error);
            return {
                success: false,
                extractedText: '',
                confidence: 0,
                method: 'Tesseract.js (Failed)',
                error: 'Failed to extract text from image: ' + error.message
            };
        }
    }

    // Extract text from PDF files
    async function extractTextFromPDF(file) {
        try {
            console.log('📑 Starting PDF text extraction...');
            
            const arrayBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
            let fullText = '';
            
            console.log('📄 PDF pages:', pdf.numPages);
            
            for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                const page = await pdf.getPage(pageNum);
                const textContent = await page.getTextContent();
                const pageText = textContent.items.map(item => item.str).join(' ');
                fullText += pageText + '\n';
                console.log(`📖 Extracted page ${pageNum}/${pdf.numPages}`);
            }
            
            console.log('✅ PDF extraction completed');
            console.log('📄 Total text length:', fullText.length);
            
            return {
                success: true,
                extractedText: fullText.trim(),
                confidence: 95, // PDFs generally have high text extraction confidence
                error: null
            };
            
        } catch (error) {
            console.error('❌ PDF Extraction Error:', error);
            return {
                success: false,
                extractedText: '',
                confidence: 0,
                error: 'Failed to extract text from PDF: ' + error.message
            };
        }
    }

    // Analyze the actual extracted text content
    async function analyzeExtractedText(extractedText, fileName, structuredFields = null) {
        if (!extractedText || extractedText.length < 10) {
            return {
                documentType: 'Empty or unreadable document',
                keyFindings: ['Document appears to be empty or text could not be extracted'],
                importantClauses: [],
                obligations: [],
                risks: [],
                summary: 'Unable to analyze - no readable text found',
                structured: structuredFields || {}
            };
        }

        const text = extractedText.toLowerCase();
        
        // Check for poor OCR quality
        const cleanText = extractedText.replace(/[^a-zA-Z0-9\s.,;:!?'"()-]/g, '');
        const readableRatio = cleanText.length / extractedText.length;
        
        if (readableRatio < 0.4) {
            return {
                documentType: 'Poor Quality OCR Result',
                keyFindings: ['Document image quality appears to be too low for reliable text extraction', 
                             'Consider uploading a clearer image or PDF version',
                             'Ensure good lighting and focus when taking photos of documents'],
                importantClauses: [],
                obligations: [],
                risks: ['Poor document quality may result in missed important information'],
                summary: `Image quality issue detected. OCR extracted ${extractedText.length} characters, but only ${Math.round(readableRatio * 100)}% appear to be readable text. For better results, please upload a clearer image with good lighting and focus.`,
                structured: structuredFields || {}
            };
        }

        // Attempt specialized parsing for Employment/Application forms
        let specializedFields = {};
        if (looksLikeEmploymentForm(extractedText)) {
            specializedFields = parseEmploymentApplicationFields(extractedText);
        }
        // Merge priority: OCR structuredFields (from server) < specialized client parser
        const mergedFields = Object.assign({}, structuredFields || {}, specializedFields);

        console.log('🔍 Starting comprehensive AI analysis...');
        
        // Run AI analysis in parallel for better performance
        const [summary, keyFindings, importantClauses, obligations, risks] = await Promise.all([
            generateTextSummary(extractedText),
            findKeyElements(extractedText),
            extractImportantClauses(extractedText),
            extractObligations(extractedText),
            identifyRisks(extractedText)
        ]);

        const analysis = {
            documentType: determineDocumentType(text, fileName, mergedFields),
            keyFindings: Array.isArray(keyFindings) ? keyFindings : [],
            importantClauses: Array.isArray(importantClauses) ? importantClauses : [],
            obligations: Array.isArray(obligations) ? obligations : [],
            risks: Array.isArray(risks) ? risks : [],
            summary: summary || 'Analysis completed but summary unavailable',
            structured: mergedFields
        };

        console.log('✅ AI analysis completed:', {
            summary: !!summary,
            keyFindings: analysis.keyFindings.length,
            clauses: analysis.importantClauses.length,
            obligations: analysis.obligations.length,
            risks: analysis.risks.length
        });

        return analysis;
    }

    // Heuristic to detect Employment/Application forms
    function looksLikeEmploymentForm(text) {
        const t = text.toLowerCase();
        const signals = [
            'employment application', 'application form', 'personal details',
            'reference #', 'basic details', 'educational qualification', 'declaration',
            'tcs employment', 'employer', 'candidate signature'
        ];
        const hits = signals.filter(s => t.includes(s)).length;
        return hits >= 2;
    }

    // Robust parser for employment/application fields
    function parseEmploymentApplicationFields(raw) {
        const text = normalizeSpaces(raw);
        const fields = {};
        const add = (key, val) => { if (val) fields[key] = { value: val.trim() }; };

        // Reference number
        add('Reference #', match1(text, /Reference\s*[#:\-]?\s*([A-Z]{1,3}\d{6,}|DT\d{6,}|\d{8,})/i));

        // College/Institute
        add('College/Institute', match1(text, /(College\s*\/\s*Institute|College\s*\/?\s*Institute)\s*[:\-]?\s*([^\n]+?)(?:\s{2,}|Name\s*:|Father|Date\s*of\s*Birth)/i));
        if (!fields['College/Institute']) add('College/Institute', match1(text, /College\s*[:\-]?\s*([^\n]+?)(?:\s{2,}|Name\s*:|Father)/i));

        // Name and Father's Name
        add('Name', match1(text, /\bName\b\s*[:\-]?\s*(Mr\.?|Ms\.?|Mrs\.)?\s*([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)/i));
        add("Father's Name", match1(text, /(Father'?s\s*Name|Father\s*Name)\s*[:\-]?\s*([^\n]+?)(?:\s{2,}|Date\s*of\s*Birth|DOB|Gender|Email)/i));

        // DOB
        add('Date of Birth', normalizeDate(match1(text, /(Date\s*Of\s*Birth|DOB)\s*[:\-]?\s*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})/i)));

        // Gender
        add('Gender', match1(text, /(Gender)\s*[:\-]?\s*(Male|Female|Other)/i));

        // Email and Phone
        add('Email', match1(text, /([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})/i));
        add('Phone', cleanPhone(match1(text, /(Mobile|Phone|Contact)\s*No\.?\s*[:\-]?\s*([+]?\d[\d\-() ]{6,})/i)));

        // Address (best-effort: capture a few lines after Address:)
        add('Address', match1Multiline(text, /(Current\s*Address|Present\s*Address|Address)\s*[:\-]?\s*([\s\S]{10,200}?)(?=Permanent\s*Address|Email|Phone|Mobile|\n\s*\n|Declaration)/i));

        // Education (one-liner best effort)
        add('Qualification', match1(text, /(Highest\s*Qualification|Qualification)\s*[:\-]?\s*([^\n]+?)(?:\s{2,}|Year|Percentage|CGPA)/i));

        return fields;
    }

    function normalizeSpaces(s) { return s.replace(/\r/g, '').replace(/\t/g, ' ').replace(/\u00A0/g, ' ').replace(/ +/g, ' ').replace(/\n +/g, '\n').trim(); }
    function match1(s, re) { const m = s.match(re); return m ? (m[2] || m[1]) : ''; }
    function match1Multiline(s, re) { const m = s.match(re); return m ? (m[2] || m[1]) : ''; }
    function normalizeDate(s) { if (!s) return ''; const m = s.match(/(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})/); if (!m) return s; const d=m[1].padStart(2,'0'); const mo=m[2].padStart(2,'0'); let y=m[3]; if (y.length===2) y = (parseInt(y,10)>50?'19':'20')+y; return `${d}/${mo}/${y}`; }
    function cleanPhone(s) { if (!s) return ''; const m = s.match(/[+]?[0-9][0-9\-() ]{6,}/); return m? m[0].replace(/[^+0-9]/g,''):''; }

    // Determine document type from actual content (generalized)
    function determineDocumentType(text, fileName = '', structuredFields = null) {
        const t = (text || '').toLowerCase();
        const name = (fileName || '').toLowerCase();

        // Strong signals from structured fields
        if (structuredFields && Object.keys(structuredFields).length) {
            const keys = Object.keys(structuredFields);
            const idCardSignals = ["Guardian's Name", 'Department', 'Batch', 'Valid Upto', 'Contact No', 'ID'];
            const formSignals = ['Reference #', 'College/Institute', "Father's Name", 'Date of Birth'];
            if (idCardSignals.filter(k => keys.includes(k)).length >= 2) return 'Student ID Card';
            if (formSignals.filter(k => keys.includes(k)).length >= 2) return 'Employment Document';
        }

        // Filename hints
        if (/resume|cv/.test(name)) return 'Resume/CV';
        if (/invoice|bill/.test(name)) return 'Invoice/Bill';
        if (/offer[_\- ]?letter/.test(name)) return 'Offer Letter';
        if (/statement/.test(name)) return 'Bank Statement';
        if (/id|identity|passport|aadhaar|pan|driver/.test(name)) return 'Identity Document';

        // Content-based heuristics
        const indicators = [
            { type: 'Contract/Agreement', keys: ['agreement', 'contract', 'terms and conditions', 'whereas', 'party of the first part'] },
            { type: 'Lease Agreement', keys: ['lease', 'tenant', 'landlord', 'rent', 'security deposit', 'premises'] },
            { type: 'Employment Document', keys: ['employment application', 'application form', 'employee id', 'employer', 'salary', 'designation', 'candidate declaration', 'personal details'] },
            { type: 'Non-Disclosure Agreement', keys: ['non-disclosure', 'nda', 'confidential information', 'proprietary', 'trade secret'] },
            { type: 'Terms of Service', keys: ['terms of service', 'acceptable use', 'privacy policy', 'service provider', 'user agreement'] },
            { type: 'Invoice/Bill', keys: ['invoice', 'amount due', 'payment terms', 'bill to', 'gst', 'subtotal', 'balance due'] },
            { type: 'Bank Statement', keys: ['account number', 'statement period', 'transaction', 'debit', 'credit', 'balance brought'] },
            { type: 'Court Order/Notice', keys: ['in the court of', 'petitioner', 'respondent', 'case no', 'hereby ordered'] },
            { type: 'Academic Transcript', keys: ['transcript', 'semester', 'cgpa', 'sgpa', 'subject code', 'marks obtained'] },
            { type: 'Identity Document', keys: ['id card', 'date of birth', 'blood group', 'valid upto', 'identity', 'registrar'] },
        ];

        for (const { type, keys } of indicators) {
            const matches = keys.filter(k => t.includes(k)).length;
            if (matches >= 2) return type;
        }

        // Specific tie-breakers
        if (t.includes('student') && (t.includes('id card') || t.includes('batch') || t.includes('guardian'))) {
            return 'Student ID Card';
        }

        return 'General Legal/Document Text';
    }

    // Find key elements in the text using AI
    async function findKeyElements(text) {
        try {
            if (!text || text.length < 20) return [];
            
            const prompt = `Identify the most important key elements, facts, or pieces of information in this document:

${text.substring(0, 600)}

List only the most significant elements as bullet points.`;

            console.log('🔍 Finding key elements with Gemini...');
            const result = await callGeminiAPI(prompt);
            
            if (result && result.includes('•')) {
                return result.split('•').filter(element => element.trim().length > 10).map(element => element.trim()).slice(0, 5);
            }
            
            // Basic fallback - extract key patterns
            const keyPatterns = [
                /\b[A-Z][^.]{20,80}\./g, // Capitalized sentences
                /important[^.]{10,80}\./gi,
                /note[^.]{10,80}\./gi,
                /\d+\s*(?:years?|months?|days?|%|USD|dollars?)[^.]{0,50}\./gi // Numbers with units
            ];
            
            const elements = [];
            for (const pattern of keyPatterns) {
                const matches = text.match(pattern);
                if (matches) elements.push(...matches.slice(0, 2));
                if (elements.length >= 4) break;
            }
            
            return elements.length > 0 ? elements : [];
        } catch (error) {
            console.error('❌ Key element extraction failed:', error);
            return [];
        }
    }

    // Extract important clauses from the text using AI
    async function extractImportantClauses(text) {
        try {
            if (!text || text.length < 20) return [];
            
            const prompt = `Identify important legal clauses, terms, or significant statements in this document. Return only the most important ones as a list:

${text.substring(0, 600)}

Format as bullet points of key clauses.`;

            console.log('⚖️ Extracting clauses with Gemini...');
            const result = await callGeminiAPI(prompt);
            
            if (result && result.includes('•')) {
                return result.split('•').filter(clause => clause.trim().length > 10).map(clause => clause.trim()).slice(0, 5);
            }
            
            // Basic fallback - look for legal keywords
            const legalPatterns = [
                /hereby agree[^.]{0,100}\./gi,
                /shall [^.]{10,80}\./gi,
                /terms and conditions[^.]{0,100}\./gi,
                /agreement[^.]{10,80}\./gi,
                /liability[^.]{10,80}\./gi
            ];
            
            const clauses = [];
            for (const pattern of legalPatterns) {
                const matches = text.match(pattern);
                if (matches) clauses.push(...matches.slice(0, 2));
                if (clauses.length >= 3) break;
            }
            
            return clauses.length > 0 ? clauses : [];
        } catch (error) {
            console.error('❌ Clause extraction failed:', error);
            return [];
        }
    }

    // Extract obligations from the text using AI
    async function extractObligations(text) {
        try {
            if (!text || text.length < 20) return [];
            
            const prompt = `Identify any obligations, requirements, or duties mentioned in this document. List only clear obligations:

${text.substring(0, 600)}

Format as bullet points of specific obligations.`;

            console.log('📋 Extracting obligations with Gemini...');
            const result = await callGeminiAPI(prompt);
            
            if (result && result.includes('•')) {
                return result.split('•').filter(obligation => obligation.trim().length > 10).map(obligation => obligation.trim()).slice(0, 4);
            }
            
            // Basic fallback - look for obligation keywords
            const obligationPatterns = [
                /must [^.]{10,80}\./gi,
                /shall [^.]{10,80}\./gi,
                /required to [^.]{10,80}\./gi,
                /responsible for [^.]{10,80}\./gi,
                /agree to [^.]{10,80}\./gi
            ];
            
            const obligations = [];
            for (const pattern of obligationPatterns) {
                const matches = text.match(pattern);
                if (matches) obligations.push(...matches.slice(0, 2));
                if (obligations.length >= 3) break;
            }
            
            return obligations.length > 0 ? obligations : [];
        } catch (error) {
            console.error('❌ Obligation extraction failed:', error);
            return [];
        }
    }

    // Identify risks mentioned in the text using AI
    async function identifyRisks(text) {
        try {
            if (!text || text.length < 20) return [];
            
            const prompt = `Identify potential risks, warnings, or concerns mentioned in this document. Look for risk factors:

${text.substring(0, 600)}

Format as bullet points of specific risks or concerns.`;

            console.log('⚠️ Identifying risks with Gemini...');
            const result = await callGeminiAPI(prompt);
            
            if (result && result.includes('•')) {
                return result.split('•').filter(risk => risk.trim().length > 10).map(risk => risk.trim()).slice(0, 4);
            }
            
            // Basic fallback - look for risk keywords
            const riskPatterns = [
                /risk[^.]{10,80}\./gi,
                /liability[^.]{10,80}\./gi,
                /penalty[^.]{10,80}\./gi,
                /breach[^.]{10,80}\./gi,
                /violation[^.]{10,80}\./gi,
                /warning[^.]{10,80}\./gi
            ];
            
            const risks = [];
            for (const pattern of riskPatterns) {
                const matches = text.match(pattern);
                if (matches) risks.push(...matches.slice(0, 2));
                if (risks.length >= 3) break;
            }
            
            return risks.length > 0 ? risks : [];
        } catch (error) {
            console.error('❌ Risk identification failed:', error);
            return [];
        }
    }

    // Generate a summary of the text using Gemini AI
    async function generateTextSummary(text) {
        try {
            if (!text || text.length < 20) {
                return 'Document appears to be empty or too short to summarize';
            }
            
            // Check if text is mostly garbage characters (OCR quality issue)
            const cleanText = text.replace(/[^a-zA-Z0-9\s.,;:!?'"()-]/g, '');
            const readableRatio = cleanText.length / text.length;
            
            if (readableRatio < 0.3 || text.includes('——————') || text.match(/[^a-zA-Z0-9\s.,;:!?'"()-]{10,}/)) {
                return `This document appears to have poor image quality for text extraction. The OCR detected ${text.length} characters, but most appear to be symbols or unclear text rather than readable content.

**Recommendation:** Try uploading a clearer image with better lighting and focus, or a PDF version if available. For best results, ensure the text is clearly visible and the image is high contrast.`;
            }
            
            // Use Gemini AI for summarization
            const prompt = `Please provide a concise 2-3 sentence summary of this document content, focusing on the main purpose and key information:

${text.substring(0, 800)}`;

            console.log('📝 Generating summary with Gemini...');
            const summary = await callGeminiAPI(prompt);
            
            if (summary && summary.length > 10) {
                return summary.trim();
            }
            
            // Fallback to basic extraction if Gemini fails
            const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);
            if (sentences.length >= 2) {
                return sentences.slice(0, 2).join('. ').trim() + '.';
            }
            
            return 'Document content identified but detailed summary unavailable';
        } catch (error) {
            console.error('❌ Summary generation failed:', error);
            // Simple fallback
            const firstLine = text.split('\n')[0];
            return firstLine.length > 5 ? `Document contains: ${firstLine.substring(0, 100)}...` : 'Summary generation temporarily unavailable';
        }
    }

    // Create new chat
    function createNewChat() {
        console.log('Creating new chat...');
        const chatId = 'chat_' + Date.now();
        const chat = {
            id: chatId,
            title: 'New Chat',
            messages: [],
            createdAt: new Date()
        };
        
        chatHistory.unshift(chat);
        currentChatId = chatId;
        
        console.log('📝 Created new chat:', chatId);
        updateHistoryDisplay();
        return chat;
    }
    
    // Update sidebar history
    function updateHistoryDisplay() {
        console.log('📋 Updating history display with', chatHistory.length, 'chats');
        historyList.innerHTML = '';
        
        if (chatHistory.length === 0) {
            historyList.innerHTML = '<div style="color: rgba(255,255,255,0.5); padding: 1rem; text-align: center; font-size: 0.85rem;">No recent chats</div>';
            return;
        }
        
        chatHistory.forEach(chat => {
            const item = document.createElement('div');
            item.className = 'history-item';
            if (chat.id === currentChatId) {
                item.classList.add('active');
            }
            
            // Create chat title element
            const titleElement = document.createElement('span');
            titleElement.className = 'chat-title';
            titleElement.textContent = chat.title;
            titleElement.style.cssText = `
                flex: 1;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            `;
            
            // Create delete button
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-chat-btn';
            deleteBtn.innerHTML = '×';
            deleteBtn.style.cssText = `
                background: rgba(255, 59, 48, 0.8);
                border: none;
                color: white;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                font-size: 14px;
                font-weight: bold;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
                opacity: 0;
                margin-left: 8px;
                flex-shrink: 0;
            `;
            
            // Make item a flex container
            item.style.cssText = `
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
            `;
            
            // Show delete button on hover
            item.addEventListener('mouseenter', () => {
                deleteBtn.style.opacity = '1';
            });
            
            item.addEventListener('mouseleave', () => {
                deleteBtn.style.opacity = '0';
            });
            
            // Delete button hover effect
            deleteBtn.addEventListener('mouseenter', () => {
                deleteBtn.style.background = 'rgba(255, 59, 48, 1)';
                deleteBtn.style.transform = 'scale(1.1)';
            });
            
            deleteBtn.addEventListener('mouseleave', () => {
                deleteBtn.style.background = 'rgba(255, 59, 48, 0.8)';
                deleteBtn.style.transform = 'scale(1)';
            });
            
            // Delete chat handler
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent chat selection
                deleteChat(chat.id);
            });
            
            // Load chat on title click
            titleElement.addEventListener('click', () => loadChat(chat.id));
            
            item.appendChild(titleElement);
            item.appendChild(deleteBtn);
            historyList.appendChild(item);
            console.log('📋 Added history item with delete button:', chat.title);
        });
    }
    
    // Delete a specific chat
    function deleteChat(chatId) {
        console.log('🗑️ Deleting chat:', chatId);
        
        // Find chat index
        const chatIndex = chatHistory.findIndex(c => c.id === chatId);
        if (chatIndex === -1) {
            console.warn('Chat not found for deletion:', chatId);
            return;
        }
        
        // Remove chat from history
        const deletedChat = chatHistory.splice(chatIndex, 1)[0];
        console.log('🗑️ Deleted chat:', deletedChat.title);
        
        // If this was the active chat, switch to another or create new
        if (currentChatId === chatId) {
            if (chatHistory.length > 0) {
                // Load the most recent remaining chat
                loadChat(chatHistory[0].id);
            } else {
                // No chats left, create a completely fresh new chat
                console.log('🆕 Creating fresh chat after deleting last chat');
                
                // Clear the chat window completely
                chatWindow.innerHTML = '';
                
                // Reset current chat ID
                currentChatId = null;
                
                // Create a brand new chat
                createNewChat();
                
                // Add fresh welcome message
                addMessage(WELCOME_TEXT, 'ai');
                
                console.log('✅ Fresh chat created successfully');
            }
        }
        
        // Update the history display
        updateHistoryDisplay();
    }
    
    // Load specific chat
    function loadChat(chatId) {
        console.log('Loading chat:', chatId);
        const chat = chatHistory.find(c => c.id === chatId);
        if (!chat) return;
        
        currentChatId = chatId;
        chatWindow.innerHTML = '';
        
        if (chat.messages.length === 0) {
            addMessage(WELCOME_TEXT, 'ai');
        } else {
            chat.messages.forEach(msg => {
                // Create a mock file object for preview if file info exists
                let fileForPreview = null;
                if (msg.file) {
                    fileForPreview = {
                        name: msg.file.name,
                        type: msg.file.type,
                        size: msg.file.size,
                        // Note: This is just metadata - actual file content is not stored
                        isStoredFile: true
                    };
                }
                addMessage(msg.content, msg.sender, fileForPreview);
            });
        }
        
        updateHistoryDisplay();
    }
    
    // Simple Markdown to HTML converter for basic formatting
    function convertMarkdownToHTML(text) {
        if (!text) return text;
        
        return text
            // Convert **bold** to <strong>bold</strong>
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            // Convert *italic* to <em>italic</em>
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            // Convert ## Heading to <h2>Heading</h2>
            .replace(/^## (.+$)/gm, '<h2>$1</h2>')
            // Convert # Heading to <h1>Heading</h1>
            .replace(/^# (.+$)/gm, '<h1>$1</h1>')
            // Convert line breaks to <br>
            .replace(/\n/g, '<br>');
    }
    
    // Add message to window
    function addMessage(content, sender, file = null) {
        console.log('Adding message:', sender, content.substring(0, 50) + '...', file ? 'with file' : '');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // If there's a file, create preview first
        if (file && sender === 'user') {
            const filePreview = createFilePreview(file);
            contentDiv.appendChild(filePreview);
            
            // Add text content if exists
            if (content && !content.startsWith('[File uploaded:')) {
                const textDiv = document.createElement('div');
                textDiv.className = 'message-text';
                textDiv.style.marginTop = '0.5rem';
                textDiv.textContent = content;
                contentDiv.appendChild(textDiv);
            }
        } else {
            // For AI messages, convert Markdown and use innerHTML
            // For user messages, use textContent for security
            if (sender === 'ai') {
                contentDiv.innerHTML = convertMarkdownToHTML(content);
            } else {
                contentDiv.textContent = content;
            }
        }
        
        messageDiv.appendChild(contentDiv);
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        
        console.log('✅ Message added successfully');
    }
    
    // Create file preview element
    function createFilePreview(file) {
        const previewContainer = document.createElement('div');
        previewContainer.className = 'file-preview';
        previewContainer.style.cssText = `
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            position: relative;
            backdrop-filter: blur(10px);
            max-width: 300px;
        `;
        
        // Check if it's an image and we have actual file data (not just metadata)
        if (file.type && file.type.startsWith('image/') && !file.isStoredFile) {
            const img = document.createElement('img');
            img.style.cssText = `
                max-width: 100%;
                max-height: 200px;
                border-radius: 8px;
                object-fit: cover;
                display: block;
            `;
            
            // Create object URL for preview
            const objectUrl = URL.createObjectURL(file);
            img.src = objectUrl;
            img.onload = () => URL.revokeObjectURL(objectUrl); // Clean up after loading
            
            previewContainer.appendChild(img);
        } else {
            // For non-image files or stored files, show file icon and info
            const fileIcon = document.createElement('div');
            fileIcon.style.cssText = `
                font-size: 3rem;
                text-align: center;
                margin-bottom: 0.5rem;
                opacity: 0.7;
            `;
            
            // Choose icon based on file type
            if (file.type && file.type.startsWith('image/')) {
                fileIcon.textContent = '🖼️';
            } else if (file.type && file.type.includes('pdf')) {
                fileIcon.textContent = '📄';
            } else {
                fileIcon.textContent = '📁';
            }
            
            previewContainer.appendChild(fileIcon);
            
            // Add "Previously uploaded" indicator for stored files
            if (file.isStoredFile) {
                const storedIndicator = document.createElement('div');
                storedIndicator.style.cssText = `
                    font-size: 0.7rem;
                    color: rgba(255, 255, 255, 0.5);
                    text-align: center;
                    margin-bottom: 0.5rem;
                    font-style: italic;
                `;
                storedIndicator.textContent = 'Previously uploaded';
                previewContainer.appendChild(storedIndicator);
            }
        }
        
        // File info
        const fileInfo = document.createElement('div');
        fileInfo.style.cssText = `
            font-size: 0.85rem;
            color: rgba(255, 255, 255, 0.8);
            text-align: center;
            margin-top: 0.5rem;
        `;
        
        const fileName = document.createElement('div');
        fileName.style.fontWeight = '500';
        fileName.textContent = file.name;
        
        const fileSize = document.createElement('div');
        fileSize.style.cssText = `
            font-size: 0.75rem;
            opacity: 0.7;
            margin-top: 0.2rem;
        `;
        fileSize.textContent = formatFileSize(file.size);
        
        fileInfo.appendChild(fileName);
        fileInfo.appendChild(fileSize);
        previewContainer.appendChild(fileInfo);
        
        return previewContainer;
    }
    
    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Save message to current chat
    function saveMessage(content, sender, file = null) {
        if (!currentChatId) return;
        
        const chat = chatHistory.find(c => c.id === currentChatId);
        if (chat) {
            const messageData = { content, sender, timestamp: new Date() };
            
            // Store file information (but not the actual file object for memory efficiency)
            if (file) {
                messageData.file = {
                    name: file.name,
                    type: file.type,
                    size: file.size
                };
            }
            
            chat.messages.push(messageData);
            
            // Update title with first user message
            if (sender === 'user' && chat.title === 'New Chat') {
                const titleText = file ? file.name : content;
                chat.title = titleText.length > 30 ? titleText.substring(0, 30) + '...' : titleText;
                updateHistoryDisplay();
            }
        }
    }

    async function classifyLegalDocument(text) {
        try {
            const prompt = `Please classify this legal document and identify its type and key legal concepts. Focus on determining whether it's a contract, agreement, legal notice, court filing, statute, regulation, or other legal document type. Provide a concise classification:\n\n${text.substring(0, 1000)}`;
            const result = await callGeminiAPI(prompt);
            return result || 'Unable to classify document';
        } catch (error) {
            console.error('❌ Legal document classification failed:', error);
            return 'Document classification unavailable';
        }
    }

    async function summarizeLegalText(text) {
        try {
            const textToSummarize = text.length > 1000 ? text.substring(0, 1000) + '...' : text;
            const prompt = `Please provide a concise summary of this legal document, highlighting the main points, key terms, parties involved, and important legal implications:\n\n${textToSummarize}`;
            const result = await callGeminiAPI(prompt);
            return result || 'Unable to summarize document';
        } catch (error) {
            console.error('❌ Legal text summarization failed:', error);
            return 'Document summarization unavailable';
        }
    }

    async function searchCaseLaw(query, limit = 3) {
        try {
            console.log(`⚖️ Searching case law for: ${query}`);
            const searchQuery = encodeURIComponent(query);
            const response = await fetch(`${COURT_LISTENER_API}search/?type=o&q=${searchQuery}&order_by=dateFiled%20desc&stat_Precedential=on`);
            
            if (!response.ok) {
                throw new Error(`Court Listener API error: ${response.status}`);
            }
            
            const data = await response.json();
            const cases = data.results?.slice(0, limit) || [];
            
            return cases.map(legalCase => ({
                title: legalCase.caseName || 'Unknown Case',
                court: legalCase.court || 'Unknown Court',
                date: legalCase.dateFiled || 'Unknown Date',
                url: `https://www.courtlistener.com${legalCase.absolute_url}` || '#'
            }));
        } catch (error) {
            console.error('❌ Case law search failed:', error);
            return [];
        }
    }

    // Enhanced AI response with Gemini
    async function generateAIResponseWithGemini(input, documentContext = '') {
        try {
            console.log('🧠 Generating AI response with Gemini...');
            
            // Create legal-focused prompt
            const legalPrompt = `You are Lexi-Core, a professional legal assistant. Provide accurate legal guidance for this query.

${documentContext ? `Document Context: ${documentContext.substring(0, 500)}` : ''}

Legal Query: ${input}

Please provide a helpful legal response, noting when users should consult with a qualified attorney:`;

            // Use Gemini API as primary
            console.log('🧠 Using Gemini AI for legal analysis...');
            const geminiResponse = await callGeminiAPI(legalPrompt, documentContext);
            
            if (geminiResponse && geminiResponse.length > 50) {
                // Search for relevant case law
                const caseLaw = await searchCaseLaw(input, 2);
                let response = geminiResponse;
                
                if (caseLaw.length > 0) {
                    response += `\n\n📚 **Relevant Case Law:**\n`;
                    caseLaw.forEach(legalCase => {
                        response += `• **${legalCase.title}** (${legalCase.court}, ${legalCase.date})\n`;
                    });
                }
                
                response += `\n\n*⚖️ For specific legal advice, please consult with a qualified attorney.*`;
                return response;
            }
            
            // Final fallback to rule-based
            console.log('⚠️ Gemini AI failed, using rule-based fallback...');
            return await generateResponse(input, documentContext);
            
        } catch (error) {
            console.error('❌ Error generating AI response:', error);
            return await generateResponse(input, documentContext); // Fallback to rule-based with context
        }
    }

    // Gemini API functions (now fallback)
    async function callGeminiAPI(prompt, context = '') {
        try {
            console.log('🚀 Calling Gemini API...');
            const systemPrompt = `You are Lexi-Core, a professional legal assistant AI. You specialize in legal document analysis, contract review, legal advice, and helping users understand legal concepts. Always provide accurate, helpful, and professional legal guidance while noting when users should consult with a qualified attorney for specific legal matters.

${context ? `Context: ${context}` : ''}

User Query: ${prompt}`;

            const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    contents: [{
                        parts: [{
                            text: systemPrompt
                        }]
                    }],
                    generationConfig: {
                        temperature: 0.7,
                        maxOutputTokens: 1000,
                        topP: 0.8,
                        topK: 40
                    }
                })
            });

            console.log('📡 API Response Status:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('❌ API Error Response:', errorText);
                throw new Error(`API request failed: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            console.log('📝 API Response Data:', data);
            
            if (data.candidates && data.candidates[0] && data.candidates[0].content) {
                const result = data.candidates[0].content.parts[0].text;
                console.log('✅ Gemini Response Success:', result.substring(0, 100) + '...');
                return result;
            } else {
                console.error('❌ Invalid API response structure:', data);
                throw new Error('Invalid response format from API');
            }
        } catch (error) {
            console.error('❌ Gemini API Error:', error);
            return null;
        }
    }

    async function generateAIResponse(input, documentContext = '') {
        try {
            console.log('🤖 Generating AI response...');
            
            // Quick check for simple greetings - respond immediately
            const lowerInput = input.toLowerCase().trim();
            const simpleGreetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'];
            
            if (simpleGreetings.some(greeting => lowerInput === greeting || lowerInput === greeting + '!')) {
                return "Hello! I'm Lexi-Core, your AI legal assistant. How can I help you with your legal questions today?";
            }
            
            // For other queries, use Gemini AI
            return await generateAIResponseWithGemini(input, documentContext);
        } catch (error) {
            console.error('❌ Error generating AI response:', error);
            return await generateResponse(input, documentContext); // Fallback to rule-based with context
        }
    }

    // Generate AI response with document context awareness
    async function generateResponse(input, documentContext = '') {
        const lowerInput = input.toLowerCase();
        
        // If we have document context, provide document-specific analysis
        if (documentContext && documentContext.length > 50) {
            return await generateDocumentAnalysis(documentContext);
        }
        
        // Regular query responses
        if (lowerInput.includes('contract')) return responses.contract;
        if (lowerInput.includes('lawsuit') || lowerInput.includes('sue')) return responses.lawsuit;
        if (lowerInput.includes('copyright')) return responses.copyright;
        if (lowerInput.includes('hello') || lowerInput.includes('hi')) return "Hello! I'm here to help with your legal questions.";
        return responses.default + " What specific legal topic interests you?";
    }
    
    // Enhanced document analysis with real text extraction
    async function generateFileResponse(file) {
        console.log('🔄 Starting comprehensive document analysis...');
        
        try {
            // Extract actual text from the document
            const textResult = await extractDocumentText(file);
            
            if (!textResult.success) {
                const errorResponse = `❌ **DOCUMENT ANALYSIS ERROR**

**File:** "${file.name}"
**Error:** ${textResult.error}

**Alternative Analysis:**
While I couldn't extract the text content, I can still provide general guidance about this type of document based on the filename and format.

Would you like me to provide general legal guidance for this document type, or would you prefer to try uploading the document in a different format?`;
                
                return errorResponse;
            }

            // Analyze the extracted text
            const analysis = await analyzeExtractedText(textResult.extractedText, file.name, textResult.structuredFields);
            // Prefer structured fields from analysis (client parser merged), fallback to server
            const fieldsSource = (analysis.structured && Object.keys(analysis.structured).length)
                ? analysis.structured
                : (textResult.structuredFields || {});
            const displayOrder = ['Reference #','Name',"Father's Name",'College/Institute','Date of Birth','Gender','Email','Phone','Address','Qualification','Department','Batch','Contact No','Valid Upto','ID'];
            const orderedKeys = displayOrder.filter(k => fieldsSource[k]).concat(Object.keys(fieldsSource).filter(k => !displayOrder.includes(k)));
            const fieldLines = orderedKeys.map(k => `• **${k}:** ${fieldsSource[k].value}${fieldsSource[k].confidence ? ` (conf: ${fieldsSource[k].confidence}%)` : ''}`);
            
            // Generate comprehensive analysis
            const comprehensiveAnalysis = `📄 **COMPREHENSIVE DOCUMENT ANALYSIS COMPLETE**

**Document:** "${file.name}" | **Confidence:** ${textResult.confidence}%
**Text Extracted:** ${textResult.textLength || textResult.extractedText.length} characters

---

## 📋 **DOCUMENT CLASSIFICATION**
**Type Identified:** ${analysis.documentType}

## 📖 **ACTUAL DOCUMENT CONTENT**
**Document Summary:**
${analysis.summary}

${fieldLines.length ? `
## 🧾 **DETECTED FORM FIELDS**
${fieldLines.join('\n')}
` : ''}

## 🔍 **KEY FINDINGS FROM TEXT**
${analysis.keyFindings.map(finding => `• ${finding}`).join('\n')}

## ⚖️ **IMPORTANT CLAUSES FOUND**
${analysis.importantClauses.length > 0 ? 
    analysis.importantClauses.map(clause => `• ${clause}`).join('\n') : 
    '• No specific legal clauses clearly identified in extracted text'}

## 📝 **YOUR OBLIGATIONS IDENTIFIED**
${analysis.obligations.length > 0 ? 
    analysis.obligations.map(obligation => `• ${obligation}`).join('\n') : 
    '• No clear obligations found in the readable text'}

## ⚠️ **POTENTIAL RISKS DETECTED**
${analysis.risks.length > 0 ? 
    analysis.risks.map(risk => `• ${risk}`).join('\n') : 
    '• No obvious risk indicators found in extracted text'}

---

**📊 ANALYSIS STATS:**
• **Text Quality:** ${textResult.confidence > 90 ? 'Excellent' : textResult.confidence > 75 ? 'Very Good' : textResult.confidence > 60 ? 'Good' : textResult.confidence > 45 ? 'Fair' : 'Poor'} (${textResult.confidence}% confidence)${textResult.raw_confidence ? ` [Enhanced from ${textResult.raw_confidence}%]` : ''}
• **Content Length:** ${textResult.extractedText.length} characters
• **OCR Method:** ${textResult.method || 'auto'}${textResult.quality_score ? ` (quality: ${textResult.quality_score})` : ''}
• **Text Readability:** ${textResult.readable_ratio ? `${Math.round(textResult.readable_ratio * 100)}% clean text` : 'Not measured'}
• **Processing Time:** ${textResult.processing_time_ms || textResult.duration_ms || 'n/a'}${typeof textResult.processing_time_ms === 'number' ? ' ms' : ''}
• **Key Elements:** ${analysis.keyFindings.length} items found

**🔍 Would you like me to explain any specific part of what I found in your document?**`;

            // If OCR text is suspiciously short but we have fields, append raw OCR for transparency
            const rawAppendix = (textResult.extractedText.length < 30 && fieldLines.length) ? `

## 🧪 Raw OCR (short text)
${textResult.extractedText || '(empty)'}
` : '';
            
            return comprehensiveAnalysis + rawAppendix;
            
        } catch (error) {
            console.error('❌ Analysis failed:', error);
            return `❌ **ANALYSIS FAILED**

There was an error analyzing your document: ${error.message}

Please try uploading the document again or contact support if the issue persists.`;
        }
    }

    // Generate document-specific analysis when AI fails
    async function generateDocumentAnalysis(documentText) {
        try {
            // Check if the text is mostly unreadable
            if (documentText.includes('WARNING: The image quality may be too poor')) {
                return `📷 **IMAGE QUALITY ISSUE**

The uploaded image appears to have poor quality for text extraction. Here's what you can try:

**To Get Better Results:**
• **Upload a clearer image** - Ensure good lighting and focus
• **Try a different format** - PDF files usually work better than photos
• **Scan at higher resolution** - At least 300 DPI for text documents
• **Ensure text is readable** - Text should be clearly visible to the naked eye

**Alternative Options:**
• Type out specific questions about your document and I'll provide general legal guidance
• Upload a PDF version if available
• Take a new photo with better lighting and focus

*For immediate assistance, you can describe the type of legal document and ask specific questions.*`;
            }
            
            const analysis = await analyzeExtractedText(documentText, 'uploaded document');
            
            let response = `📄 **DOCUMENT ANALYSIS** (Local Processing - AI Models Currently Unavailable)

**Document Type:** ${analysis.documentType}

**Summary:** ${analysis.summary}

**Key Findings:**
${analysis.keyFindings.map(finding => `• ${finding}`).join('\n')}`;

            if (analysis.importantClauses && analysis.importantClauses.length > 0) {
                response += `

**Important Clauses:**
${analysis.importantClauses.map(clause => `• ${clause}`).join('\n')}`;
            }

            if (analysis.risks && analysis.risks.length > 0) {
                response += `

**Potential Risks:**
${analysis.risks.map(risk => `• ${risk}`).join('\n')}`;
            }

            if (analysis.obligations && analysis.obligations.length > 0) {
                response += `

**Obligations:**
${analysis.obligations.map(obligation => `• ${obligation}`).join('\n')}`;
            }

            response += `

*📝 This analysis uses local rule-based processing since advanced AI models are temporarily unavailable. For comprehensive legal advice, please consult with a qualified attorney.*`;

            return response;
            
        } catch (error) {
            console.error('Error in document analysis fallback:', error);
            return "I'm having trouble analyzing this document right now. Please try uploading it again or contact support if the issue persists.";
        }
    }
    
    // Show typing indicator (thinking dots)
    function showTyping() {
        const typing = document.createElement('div');
        typing.className = 'chat-message ai typing';
        typing.id = 'typing-indicator';
        typing.innerHTML = `
            <div class="message-content">
                <div class="thinking-dots"><span></span><span></span><span></span></div>
            </div>
        `;
        chatWindow.appendChild(typing);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return typing;
    }
    
    // Handle sending message
    function handleSend() {
        console.log('Handle send called');
        if (!textInput) {
            console.error('Text input not found');
            return;
        }
        
        const text = textInput.value.trim();
        const hasFile = attachedFile !== null;
        
        // Check if we have either text or file
        if (!text && !hasFile) {
            console.log('Empty text and no file, not sending');
            return;
        }
        
        console.log('📤 Sending:', text, hasFile ? `with file: ${attachedFile.name}` : 'text only');
        
        // Create chat if needed
        if (!currentChatId) {
            console.log('Creating new chat for message');
            createNewChat();
        }
        
        // Add user message (with or without text)
        const userMessage = text || `[File uploaded: ${attachedFile.name}]`;
        addMessage(userMessage, 'user', attachedFile);
        saveMessage(userMessage, 'user', attachedFile);
        
        // Clear input and file
        textInput.value = '';
        const currentFile = attachedFile;
        clearAttachedFile();
        
        // Show typing and respond
        const typing = showTyping();
        // Watchdog: remove typing if processing stalls
        const typingWatchdog = setTimeout(() => {
            if (document.getElementById('typing-indicator')) {
                console.warn('⏱️ Typing watchdog: removing stalled indicator');
                typing.remove();
                addMessage('Still working on your document... (network is slow). If this persists, please try again.', 'ai');
            }
        }, 30000);

        // Generate response - optimized for speed
        if (currentFile) {
            // File processing - start immediately with OCR
            (async () => {
                try {
                    const response = await generateFileResponse(currentFile);
                    if (typing) typing.remove();
                    addMessage(response, 'ai');
                    saveMessage(response, 'ai');
                } catch (e) {
                    console.error('❌ File response failed:', e);
                    if (typing) typing.remove();
                    addMessage('There was an error processing your document. Please try again.', 'ai');
                } finally {
                    clearTimeout(typingWatchdog);
                }
            })();
        } else {
            // Text messages should be fast but feel natural
            setTimeout(async () => {
                try {
                    if (typing) typing.remove();
                    const response = await generateAIResponse(text);
                    addMessage(response, 'ai');
                    saveMessage(response, 'ai');
                } finally {
                    clearTimeout(typingWatchdog);
                }
            }, 300); // Just 300ms for natural feel
        }
    }
    
    // Handle new chat with debounce protection
    let isCreatingChat = false;
    function handleNewChat() {
        if (isCreatingChat) {
            console.log('⚠️ Chat creation already in progress, ignoring...');
            return;
        }
        
        isCreatingChat = true;
        console.log('🆕 New chat requested');
        
        const chat = createNewChat();
        chatWindow.innerHTML = '';
        addMessage(WELCOME_TEXT, 'ai');
        
        console.log('✅ New chat ready');
        
        // Reset flag after a short delay
        setTimeout(() => {
            isCreatingChat = false;
        }, 500);
    }
    
    // Event listeners
    if (newChatBtn) {
        newChatBtn.addEventListener('click', handleNewChat);
        console.log('✅ New chat button connected');
    } else {
        console.warn('⚠️ New chat button not found');
    }
    
    if (sendButton && textInput) {
        sendButton.addEventListener('click', handleSend);
        textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
            }
        });
        console.log('✅ Send functionality connected');
    } else {
        console.warn('⚠️ Send button or text input not found');
    }
    
    // File upload event listeners
    if (attachFileBtn && fileInput) {
        attachFileBtn.addEventListener('click', handleFileAttachment);
        fileInput.addEventListener('change', handleFileSelect);
        console.log('✅ File upload functionality connected');
    } else {
        console.warn('⚠️ File upload elements not found');
    }

    // Toggle history functionality
    if (toggleHistoryBtn && historyContent) {
        toggleHistoryBtn.addEventListener('click', function() {
            const isCollapsed = historyContent.classList.contains('collapsed');
            
            if (isCollapsed) {
                historyContent.classList.remove('collapsed');
                toggleHistoryBtn.classList.remove('collapsed');
                console.log('📂 Chat history expanded');
            } else {
                historyContent.classList.add('collapsed');
                toggleHistoryBtn.classList.add('collapsed');
                console.log('📁 Chat history collapsed');
            }
        });
        console.log('✅ History toggle functionality connected');
    } else {
        console.warn('⚠️ History toggle elements not found');
    }
    
    // Allow clicking only on cross symbol to clear attachment
    if (attachedFileName) {
        attachedFileName.addEventListener('click', function(event) {
            // Calculate if click was in the cross symbol area (right side of the element)
            const rect = attachedFileName.getBoundingClientRect();
            const clickX = event.clientX - rect.left;
            const elementWidth = rect.width;
            
            // Cross symbol is positioned at right: 0.8rem, with width 18px
            // So if click is within the last ~30px (cross area), then clear file
            if (clickX > elementWidth - 35) {
                clearAttachedFile();
                console.log('🗑️ File cleared via cross click');
            } else {
                console.log('📝 Filename clicked but not cross symbol - file kept');
            }
        });
        
        // Set cursor to default for filename, cross will have pointer via CSS
        attachedFileName.style.cursor = 'default';
        console.log('✅ Cross-only click-to-clear connected');
    }
    
    // Initialize with first chat and welcome message
    console.log('🔧 Initializing with welcome message...');
    try {
        createNewChat();
        addMessage(WELCOME_TEXT, 'ai');
        console.log('✅ Legal Navigator ready!');
    } catch (error) {
        console.error('❌ Error during initialization:', error);
    }
    
    // Global fallback
    window.lexiNewChat = handleNewChat;
}); // Close DOMContentLoaded event listener

// CSS for thinking dots indicator
const style = document.createElement('style');
style.textContent = `
.thinking-dots { display: flex; gap: 4px; align-items: center; padding: 2px 4px; }
.thinking-dots span { width: 4px; height: 4px; background: var(--accent-color); border-radius: 50%; display: inline-block; animation: thinking 1.4s infinite both; box-shadow: 0 1px 4px rgba(0, 212, 255, 0.3); }
.thinking-dots span:nth-child(1) { animation-delay: 0s; }
.thinking-dots span:nth-child(2) { animation-delay: 0.16s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.32s; }
@keyframes thinking { 0%,80%,100%{ transform: scale(0.8); opacity: .5;} 40%{ transform: scale(1.1); opacity: 1;} }
`;
document.head.appendChild(style);
