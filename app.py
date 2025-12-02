import streamlit as st
import google.generativeai as genai
import cv2
import json
import os
import tempfile
import time
from pathlib import Path
from fpdf import FPDF
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Video to User Guide Converter",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []


def configure_gemini(api_key=None):
    """Configure Gemini API with API key from parameter or session state."""
    # Get API key from parameter, session state, or environment (in that order)
    if api_key:
        key = api_key
    elif 'google_api_key' in st.session_state and st.session_state.google_api_key:
        key = st.session_state.google_api_key
    else:
        # Fallback to environment variable
        key = os.getenv('GOOGLE_API_KEY')
    
    if not key:
        st.error("❌ API key not found. Please enter your Google API key in the sidebar.")
        st.stop()
    
    genai.configure(api_key=key)
    return genai


def get_available_model(api_key=None):
    """Get an available Gemini model that supports video."""
    genai = configure_gemini(api_key)
    
    # First, list all available models
    available_models = []
    try:
        models = genai.list_models()
        # Filter models that support generateContent
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                # Extract model name (remove 'models/' prefix if present)
                model_name = m.name.replace('models/', '')
                available_models.append(model_name)
        
        # Store in session state for debugging
        if available_models:
            st.session_state.available_models = available_models
    except Exception as e:
        st.warning(f"Could not list models: {e}")
    
    # Try models in order of preference
    model_priority = [
        'gemini-3-pro',          # Latest Gemini 3 Pro
        'gemini-3-pro-latest',   # Latest variant
        'gemini-1.5-flash',       # Usually most available
        'gemini-1.5-pro',         # Pro version
        'gemini-pro',             # Older version
        'gemini-1.5-flash-latest',
        'gemini-1.5-pro-latest',
    ]
    
    # First try from available_models list if we got it
    if available_models:
        # Prefer models with '3' and 'pro' (Gemini 3 Pro - latest)
        for model_name in available_models:
            if '3' in model_name.lower() and 'pro' in model_name.lower():
                try:
                    model = genai.GenerativeModel(model_name)
                    return model, model_name
                except Exception as e:
                    continue
        
        # Then try models with '1.5' and 'flash' (usually most available)
        for model_name in available_models:
            if '1.5' in model_name.lower() and 'flash' in model_name.lower():
                try:
                    model = genai.GenerativeModel(model_name)
                    return model, model_name
                except Exception as e:
                    continue
        
        # Then try '1.5' and 'pro'
        for model_name in available_models:
            if '1.5' in model_name.lower() and 'pro' in model_name.lower():
                try:
                    model = genai.GenerativeModel(model_name)
                    return model, model_name
                except Exception as e:
                    continue
        
        # Then any model with '3' (Gemini 3)
        for model_name in available_models:
            if '3' in model_name.lower():
                try:
                    model = genai.GenerativeModel(model_name)
                    return model, model_name
                except Exception as e:
                    continue
        
        # Then any model with '1.5'
        for model_name in available_models:
            if '1.5' in model_name.lower():
                try:
                    model = genai.GenerativeModel(model_name)
                    return model, model_name
                except Exception as e:
                    continue
        
        # Fallback to first available model
        for model_name in available_models:
            try:
                model = genai.GenerativeModel(model_name)
                return model, model_name
            except Exception as e:
                continue
    
    # If listing failed or no models worked, try common names directly
    for model_name in model_priority:
        try:
            model = genai.GenerativeModel(model_name)
            return model, model_name
        except Exception:
            continue
    
    # Final fallback - show error with available models if we have them
    error_msg = "❌ Could not find any available Gemini models."
    if available_models:
        error_msg += f"\n\nAvailable models: {', '.join(available_models[:10])}"
    st.error(error_msg)
    st.stop()


def upload_video_to_gemini(video_path, api_key=None):
    """Upload video to Gemini File API and wait for processing."""
    genai = configure_gemini(api_key)
    
    with st.spinner("📤 Uploading video to Gemini..."):
        # Upload the video file
        video_file = genai.upload_file(path=video_path)
        st.session_state.temp_files.append(video_file)  # Track for cleanup
    
    # Wait for processing
    with st.spinner("⏳ Waiting for video processing..."):
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            st.error("❌ Video processing failed. Please try again with a different video.")
            st.stop()
        
        if video_file.state.name != "ACTIVE":
            st.error(f"❌ Unexpected video state: {video_file.state.name}")
            st.stop()
    
    st.success("✅ Video processed successfully!")
    return video_file


def get_default_prompt():
    """Get the default prompt for video analysis."""
    return """You are analyzing a product explainer video designed for new users. Create a friendly, beginner-friendly step-by-step guide that helps someone understand how to use the product being demonstrated.

IMPORTANT: Focus ONLY on the actual product/application being demonstrated. DO NOT include steps that show generic applications like Microsoft Word, Excel, PowerPoint, or other standard software unless they are specifically part of the product being explained. Skip any screenshots or steps that show generic software interfaces.

Return ONLY a valid JSON array (no markdown, no code blocks, no additional text) with the following structure:

[
    {
        "timestamp_seconds": 5.2,
        "title": "Welcome to [Product Name]",
        "description": "The video begins by introducing the main dashboard and key features you'll be using.",
        "commentary": "As the narrator explains: 'Welcome! Let's start by exploring the main interface' - this sets the stage for new users."
    },
    {
        "timestamp_seconds": 12.5,
        "title": "Creating Your First Project",
        "description": "Click the 'New Project' button in the top right corner to get started.",
        "commentary": "The narrator guides you: 'To create your first project, simply click here' - making it easy for beginners to follow along."
    }
]

Requirements:
- Extract all meaningful steps that demonstrate the PRODUCT being explained (not generic software)
- Use a friendly, welcoming tone suitable for new users - like a product explainer video
- Each step must have:
  - timestamp_seconds (float/int): The exact timestamp in seconds
  - title (string): A friendly, clear step title that a new user would understand
  - description (string): What action is being performed, written in simple, beginner-friendly language
  - commentary (string): Relevant audio statements, narration, or spoken instructions that relate to what's happening. Write this in a helpful, explanatory tone that connects the audio to the visual context.
- SKIP any steps that show generic applications (Word, Excel, PowerPoint, etc.) unless they are specifically part of the product tutorial
- Focus on the unique features and workflows of the product being demonstrated
- Commentary should be written in a friendly, helpful tone - like a teacher explaining to a new user
- If there's no relevant audio at a timestamp, use an empty string for commentary
- Timestamps should be in seconds from the start of the video
- Return ONLY the JSON array, nothing else"""


def analyze_video_with_gemini(video_file, custom_prompt=None, api_key=None):
    """Send prompt to Gemini to extract step-by-step instructions."""
    genai = configure_gemini(api_key)
    
    # Use custom prompt if provided, otherwise use default
    prompt = custom_prompt if custom_prompt else get_default_prompt()

    with st.spinner("🤖 Analyzing video with Gemini..."):
        try:
            model, model_name = get_available_model(api_key)
            st.info(f"✅ Using model: {model_name}")
            response = model.generate_content([video_file, prompt])
        except Exception as e:
            st.error(f"❌ Error with model {model_name if 'model_name' in locals() else 'unknown'}: {str(e)}")
            # Try to show available models for debugging
            try:
                genai = configure_gemini(api_key)
                models = genai.list_models()
                available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                st.info(f"Available models: {', '.join(available[:5])}")  # Show first 5
            except:
                pass
            raise
    
    return response.text


def parse_steps_json(json_text):
    """Parse JSON response from Gemini."""
    try:
        # Clean the response - remove markdown code blocks if present
        json_text = json_text.strip()
        if json_text.startswith("```"):
            # Remove markdown code blocks
            lines = json_text.split('\n')
            json_text = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])
        elif json_text.startswith("```json"):
            lines = json_text.split('\n')
            json_text = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])
        
        steps = json.loads(json_text)
        if not isinstance(steps, list):
            st.error("❌ Expected a JSON array, but got a different type.")
            st.stop()
        
        # Ensure all steps have required fields, add commentary if missing
        for step in steps:
            if 'commentary' not in step:
                step['commentary'] = ''
        
        return steps
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse JSON response: {e}")
        st.error(f"Response received: {json_text[:500]}")
        st.stop()


def extract_frame_at_timestamp(video_path, timestamp_seconds, output_path):
    """Extract a frame from video at specific timestamp using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    
    # Get FPS to calculate frame number
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp_seconds * fps)
    
    # Seek to the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Save the frame
        cv2.imwrite(output_path, frame)
        return True
    return False


def generate_pdf(steps, image_paths, output_path):
    """Generate PDF with steps, images, descriptions, and commentary using FPDF."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    for i, (step, img_path) in enumerate(zip(steps, image_paths), 1):
        # Add a new page
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", "B", 16)
        # Sanitize title text for FPDF compatibility
        clean_title = ''.join(char for char in step['title'] if ord(char) < 128 and char.isprintable())
        pdf.cell(0, 10, clean_title, ln=1, align='L')
        pdf.ln(5)
        
        # Image
        if os.path.exists(img_path):
            # Get image dimensions
            img_width = pdf.w - 40  # Width with margins
            pdf.image(img_path, x=10, y=None, w=img_width)
            pdf.ln(5)
        
        # Description
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(0, 0, 0)  # Black text
        # Sanitize description text for FPDF compatibility
        clean_description = ''.join(char for char in step['description'] if ord(char) < 128 and char.isprintable())
        pdf.multi_cell(0, 8, clean_description, align='L')
        pdf.ln(3)
        
        # Commentary (if available)
        commentary = step.get('commentary', '')
        if commentary and commentary.strip():
            pdf.set_font("Arial", "I", 11)
            pdf.set_text_color(50, 50, 150)  # Dark blue for commentary
            # Remove emojis and non-ASCII characters that FPDF can't handle
            # Keep only ASCII printable characters
            clean_commentary = ''.join(char for char in commentary if ord(char) < 128 and char.isprintable())
            pdf.multi_cell(0, 7, clean_commentary, align='L')
            pdf.ln(3)
        
        # Reset text color
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
    
    pdf.output(output_path)
    return output_path


def cleanup_temp_files(api_key=None):
    """Clean up temporary files."""
    try:
        genai = configure_gemini(api_key)
    except:
        genai = None
    
    for file_obj in st.session_state.temp_files:
        # Handle Gemini uploaded files
        if hasattr(file_obj, 'name') and hasattr(file_obj, 'state') and genai:
            try:
                genai.delete_file(file_obj.name)
            except Exception as e:
                st.warning(f"Could not delete Gemini file {file_obj.name}: {e}")
        # Handle local file paths
        elif isinstance(file_obj, str) and os.path.exists(file_obj):
            try:
                os.remove(file_obj)
            except Exception as e:
                st.warning(f"Could not delete temp file {file_obj}: {e}")


def main():
    st.title("🎬 Video to User Guide Converter")
    st.markdown("Upload an MP4 video and get a step-by-step PDF user guide generated automatically!")
    
    # Sidebar for API key input
    with st.sidebar:
        st.header("⚙️ API Configuration")
        
        # API Key input
        api_key_input = st.text_input(
            "Google API Key:",
            value=st.session_state.get('google_api_key', ''),
            type="password",
            help="Enter your Google Gemini API key. Get one at https://makersuite.google.com/app/apikey",
            placeholder="Enter your API key here"
        )
        
        # Store API key in session state
        if api_key_input:
            st.session_state.google_api_key = api_key_input
            st.success("✅ API Key saved!")
        elif 'google_api_key' in st.session_state:
            # Keep existing key if input is cleared
            pass
        
        st.divider()
        
        if st.button("🔍 List Available Models"):
            if not st.session_state.get('google_api_key'):
                st.error("❌ Please enter your API key first!")
            else:
                try:
                    genai = configure_gemini()
                    models = genai.list_models()
                    available = []
                    for m in models:
                        if 'generateContent' in m.supported_generation_methods:
                            model_name = m.name.replace('models/', '')
                            available.append(model_name)
                    
                    if available:
                        st.success(f"Found {len(available)} models:")
                        for model in available[:10]:  # Show first 10
                            st.text(f"  • {model}")
                    else:
                        st.warning("No models found with generateContent support")
                except Exception as e:
                    st.error(f"Error listing models: {str(e)}")
        
        st.divider()
        st.caption("💡 Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an MP4 video file",
        type=['mp4'],
        help="Upload an MP4 video file to convert to a user guide"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
            st.session_state.temp_files.append(temp_video_path)
        
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Prompt editor section
        st.divider()
        st.subheader("📝 Customize Analysis Prompt")
        st.caption("Edit the prompt below to customize how the video is analyzed. The default prompt extracts steps with audio commentary.")
        
        # Initialize session state for prompt if not exists
        if 'custom_prompt' not in st.session_state:
            st.session_state.custom_prompt = get_default_prompt()
        
        # Text area for prompt editing
        edited_prompt = st.text_area(
            "Analysis Prompt:",
            value=st.session_state.custom_prompt,
            height=300,
            help="Modify this prompt to change how Gemini analyzes your video. Make sure to maintain the JSON structure requirements."
        )
        
        # Update session state
        st.session_state.custom_prompt = edited_prompt
        
        # Show prompt info
        with st.expander("ℹ️ About the Prompt"):
            st.markdown("""
            **Default Prompt Features:**
            - Creates a friendly, beginner-friendly product explainer guide
            - Focuses on the actual product being demonstrated (skips generic software like Word, Excel, etc.)
            - Analyzes both visual content (screen) and audio commentary
            - Uses a welcoming tone suitable for new users
            - Returns JSON with: timestamp, title, description, and commentary
            - Commentary connects audio statements to visual actions in a helpful way
            
            **Tips:**
            - Keep the JSON structure requirements in your custom prompt
            - The prompt is sent to Gemini along with your video
            - You can customize the tone or focus areas as needed
            - The default prompt filters out generic application screenshots
            """)
        
        # Process button
        if st.button("🚀 Generate User Guide", type="primary"):
            # Check if API key is set
            if not st.session_state.get('google_api_key'):
                st.error("❌ Please enter your Google API key in the sidebar first!")
                st.stop()
            
            try:
                api_key = st.session_state.google_api_key
                
                # Step 1: Configure Gemini
                genai = configure_gemini(api_key)
                
                # Step 2: Upload video to Gemini
                video_file = upload_video_to_gemini(temp_video_path, api_key)
                
                # Step 3: Analyze video with Gemini
                json_response = analyze_video_with_gemini(video_file, st.session_state.custom_prompt, api_key)
                
                # Parse JSON
                steps = parse_steps_json(json_response)
                st.success(f"✅ Extracted {len(steps)} steps from the video!")
                
                # Display steps preview
                with st.expander("📋 Preview Steps", expanded=True):
                    for i, step in enumerate(steps, 1):
                        st.markdown(f"**{step['title']}** (at {step['timestamp_seconds']}s)")
                        st.caption(step['description'])
                        # Show commentary if available
                        commentary = step.get('commentary', '')
                        if commentary and commentary.strip():
                            st.info(f"💬 **Audio Commentary:** {commentary}")
                
                # Step 4: Extract frames
                st.info("📸 Extracting screenshots from video...")
                image_paths = []
                progress_bar = st.progress(0)
                
                for idx, step in enumerate(steps):
                    timestamp = step['timestamp_seconds']
                    temp_img_path = tempfile.mktemp(suffix='.jpg')
                    st.session_state.temp_files.append(temp_img_path)
                    
                    if extract_frame_at_timestamp(temp_video_path, timestamp, temp_img_path):
                        image_paths.append(temp_img_path)
                    else:
                        st.warning(f"⚠️ Could not extract frame at {timestamp}s")
                        image_paths.append(None)
                    
                    progress_bar.progress((idx + 1) / len(steps))
                
                # Step 5: Generate PDF
                st.info("📄 Generating PDF...")
                pdf_output_path = tempfile.mktemp(suffix='.pdf')
                st.session_state.temp_files.append(pdf_output_path)
                
                # Filter out None image paths
                valid_steps = []
                valid_image_paths = []
                for step, img_path in zip(steps, image_paths):
                    if img_path is not None:
                        valid_steps.append(step)
                        valid_image_paths.append(img_path)
                
                if valid_steps:
                    generate_pdf(valid_steps, valid_image_paths, pdf_output_path)
                    st.session_state.pdf_path = pdf_output_path
                    st.success("✅ PDF generated successfully!")
                else:
                    st.error("❌ No valid frames extracted. Cannot generate PDF.")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)
        
        # Step 6: Download PDF
        if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
            st.divider()
            st.header("📥 Download Your User Guide")
            with open(st.session_state.pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=f"{Path(uploaded_file.name).stem}_user_guide.pdf",
                    mime="application/pdf",
                    type="primary"
                )
            
            # Cleanup option
            if st.button("🧹 Clean up temporary files"):
                api_key = st.session_state.get('google_api_key')
                cleanup_temp_files(api_key)
                st.session_state.pdf_path = None
                st.session_state.temp_files = []
                st.success("✅ Temporary files cleaned up!")
                st.rerun()
    
    # Cleanup on app close (this runs when session ends)
    if st.session_state.temp_files:
        # Note: This is a best-effort cleanup
        pass


if __name__ == "__main__":
    main()

