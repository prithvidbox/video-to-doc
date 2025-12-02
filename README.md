# Video to User Guide Converter

A Streamlit web application that converts MP4 videos into step-by-step PDF user guides using Google's Gemini 1.5 Pro AI model.

## Features

- 🎬 Upload MP4 video files
- 🤖 AI-powered video analysis using Gemini 1.5 Pro
- 📸 Automatic screenshot extraction at key timestamps
- 📄 Professional PDF generation with step-by-step instructions
- ⬇️ Easy download of generated user guides

## Prerequisites

- Python 3.8 or higher
- Google API key with access to Gemini 1.5 Pro

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

To get a Google API key:
- Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create a new API key
- Copy it to your `.env` file

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser to the URL shown (typically `http://localhost:8501`)

3. Upload an MP4 video file using the file uploader

4. Click "Generate User Guide" to:
   - Upload the video to Gemini
   - Analyze the video content
   - Extract step-by-step instructions
   - Capture screenshots at key timestamps
   - Generate a PDF user guide

5. Download your generated PDF user guide

## How It Works

1. **Video Upload**: The app saves your uploaded MP4 to a temporary location
2. **Gemini Processing**: Uploads the video to Google's Gemini File API and waits for processing
3. **AI Analysis**: Uses Gemini 1.5 Pro to analyze the video and extract structured step-by-step instructions
4. **Frame Extraction**: Uses OpenCV to extract screenshots at the timestamps identified by Gemini
5. **PDF Generation**: Creates a professional PDF with titles, screenshots, and descriptions for each step
6. **Download**: Provides a download button for the generated PDF

## Project Structure

```
video-txt/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example       # Example environment file (create .env from this)
├── .env               # Your API key (not in git)
└── README.md          # This file
```

## Dependencies

- `streamlit`: Web application framework
- `google-generativeai`: Google Gemini API client
- `opencv-python`: Video processing and frame extraction
- `fpdf2`: PDF generation
- `python-dotenv`: Environment variable management

## Notes

- The application handles temporary file cleanup automatically
- Video processing may take some time depending on video length
- Make sure your video is clear and contains visible steps for best results
- The Gemini API has usage limits - check your quota if you encounter errors

## Troubleshooting

**"GOOGLE_API_KEY not found"**
- Make sure you've created a `.env` file with your API key
- Check that the key is correctly formatted (no quotes, no spaces)

**"Video processing failed"**
- Ensure your video file is a valid MP4
- Check that your Gemini API key has access to the File API
- Try with a shorter video first

**"Failed to parse JSON response"**
- The AI might return unexpected format - try regenerating
- Check the error message for the actual response received

## License

This project is open source and available for personal and commercial use.

# video-to-doc
