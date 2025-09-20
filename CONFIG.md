# Configuration Template for Legal Navigator

## Required API Keys

### Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or select a project
3. Generate an API key
4. Copy the key and replace 'YOUR_GEMINI_API_KEY_HERE' in script.js

### Court Listener API (Optional)
1. Visit [Court Listener API](https://www.courtlistener.com/api/)
2. Create an account if needed
3. No API key required for basic usage

## Security Notes

- Never commit API keys to version control
- Use environment variables in production
- Keep API keys secure and rotate them regularly
- Monitor API usage to detect unauthorized access

## Local Development Setup

1. Edit `script.js` and replace:
   ```javascript
   const GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE';
   ```

2. For production, consider using environment variables:
   ```javascript
   const GEMINI_API_KEY = process.env.GEMINI_API_KEY || 'your-fallback-key';
   ```

## API Rate Limits

- **Gemini API**: Check current limits in your Google AI Studio
- **Court Listener**: Generally allows reasonable usage for research

## Troubleshooting

- **403 Errors**: Check API key validity
- **Rate Limit Errors**: Implement delay between requests
- **CORS Issues**: Ensure proper server configuration