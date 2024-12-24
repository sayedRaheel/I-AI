# AI Voice Assistant with Custom Voice Cloning

A serverless AI assistant that can process both text and audio inputs, respond using LLM (Groq), and generate responses using a SAYED RAHEEL'S custom-cloned voice. Deployed on RunPod for scalable, GPU-accelerated processing.

## Features

- **Dual Input Processing**: 
  - Text-based queries
  - Audio input (Speech-to-Text using Whisper)
  
- **Advanced Language Processing**:
  - Uses Groq's LLaMA model for intelligent responses
  - Maintains context and personality as a professional AI assistant
  
- **Custom Voice Synthesis**:
  - Uses OuteTTS for high-quality speech synthesis
  - Supports custom voice cloning (with fallback to default voices)
  - Real-time audio generation

## System Architecture

```
Input (Text/Audio) → Processing Pipeline → Response (Text + Audio)

Text Input Path:
Text → LLM Processing → TTS Generation → Audio Output

Audio Input Path:
Audio → STT (Whisper) → LLM Processing → TTS Generation → Audio Output
```

## Prerequisites

- RunPod account
- Groq API key
- Docker installed locally (for testing)
- Python 3.10+
- NVIDIA GPU support (for deployment)

## Environment Variables

Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Installation & Local Testing

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-voice-assistant.git
cd ai-voice-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run local tests:
```bash
python handler.py
```

## Docker Build & Deployment

1. Build Docker image:
```bash
docker build -t ai-voice-assistant .
```

2. Test locally:
```bash
docker run -p 8000:8000 --env-file .env ai-voice-assistant
```

3. Deploy to RunPod:
- Push to GitHub
- Connect GitHub to RunPod
- Deploy as serverless endpoint

## API Reference

### Text Input
```json
POST /
{
    "input": {
        "type": "text",
        "text": "Your question here"
    }
}
```

### Audio Input
```json
POST /
{
    "input": {
        "type": "audio",
        "audio": "base64_encoded_audio"
    }
}
```

### Response Format
```json
{
    "user_input": {
        "type": "text|audio",
        "text": "transcribed_or_original_text"
    },
    "assistant_response": {
        "text": "ai_generated_response",
        "audio": "base64_encoded_audio"
    }
}
```

## Voice Profile Management

- Default location: `sayed_voice.json`
- Fallback to default male voice if custom profile fails
- Support for different voice profiles (modify `handler.py`)

## Error Handling

The system handles various error scenarios:
- Failed voice profile loading
- Audio transcription errors
- LLM API timeouts
- TTS generation issues

## Performance Considerations

- GPU acceleration for TTS
- Serverless scaling on RunPod
- Temporary file cleanup
- Memory management

## Monitoring & Logging

- Built-in health checks
- Detailed logging for debugging
- RunPod dashboard monitoring

## Security

- API key management via environment variables
- Temporary file handling
- Request validation

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Your chosen license]

## Contact

[Your contact information]

## Acknowledgments

- RunPod for serverless infrastructure
- Groq for LLM API
- OuteTTS for voice synthesis