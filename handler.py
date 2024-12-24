import runpod
import asyncio
import base64
import outetts
import torch
from groq import Groq
import os

# Initialize TTS and Groq clients

def setup_tts():
    model_config = outetts.HFModelConfig_v1(
        model_path="OuteAI/OuteTTS-0.2-500M",
        language="en",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)
    return interface

async def async_handler(job):
    try:
        # Initialize clients
        groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        tts_interface = setup_tts()
        
        # Load voice profile
        try:
            speaker = tts_interface.load_speaker("sayed_voice.json")
        except:
            speaker = tts_interface.load_default_speaker(name="male_1")
        
        # Get input from job
        input_type = job["input"]["type"]
        
        if input_type == "text":
            text_input = job["input"]["text"]
        else:  # audio input
            audio_base64 = job["input"]["audio"]
            audio_bytes = base64.b64decode(audio_base64)
            
            # Save temporary file
            temp_filename = "/tmp/temp_recording.wav"
            with open(temp_filename, "wb") as f:
                f.write(audio_bytes)
                
            # Transcribe using Groq
            with open(temp_filename, "rb") as file:
                translation = groq_client.audio.translations.create(
                    file=(temp_filename, file.read()),
                    model="whisper-large-v3",
                    response_format="json",
                    temperature=0.0
                )
            text_input = translation.text

        # Get LLM response
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are Sayed Raheel Hussain's AI assistant. Provide concise, professional responses based on these facts:

Professional Background:
- ML Engineer & AI Researcher based in New Jersey
- Contact: sayedraheel1995@gmail.com, 551-998-3741
- Portfolio: https://sayedraheel.me
- LinkedIn: https://linkedin.com/in/sayedraheel
- GitHub: https://github.com/sayedraheel

Research Publications & Links:
- BovineTeatNet (2024): [Paper]: https://arxiv.org/abs/bovine-teat-net
- VetMedGPT (2024): [Paper]: https://arxiv.org/abs/vet-med-gpt
- LungNet (2023): [Paper]: https://arxiv.org/abs/lung-net-segmentation

Major Projects & Links:
1. LlamaSearch: https://github.com/sayedraheel/llama-search
   - Multiagent search system with Llama 70B
   - 500k tokens/sec via Groq integration

2. Medical Chatbot: https://github.com/sayedraheel/med-chat
   - RAG-enhanced LLM using Llama 7B
   - Sub-second retrieval with FAISS



When sharing links:
- Only share these verified links when specifically asked
- Provide context about the project/paper along with the link
- For papers, mention key findings/metrics
- For projects, highlight main features and technologies used

Professional Background:
- ML Engineer & AI Researcher based in New Jersey
- Contact: sayedraheel1995@gmail.com, 551-998-3741
- Currently ML Engineer at Sync AI (NY) developing image recognition for food calories/nutrients using CNNs
- Previously Data Scientist at Palazzo.ai and Research Assistant at Yeshiva University

Education:
- M.S. in Artificial Intelligence from Yeshiva University (2024), GPA: 3.85/4.0
- B.Tech. in Automation & Mechanical Engineering from Amity School of Engineering (2017), GPA: 3.5/4.0

Key Achievements:
- Published research: BovineTeatNet (mastitis detection), VetMedGPT (veterinary LLM), LungNet (cancer segmentation)
- Developed LlamaSearch: Open-source multiagent search system using Llama 70B
- Created medical chatbot using Llama 7B with RAG
- Accelerated chatbot throughput 10x using Mistral LLM with Groq
- Implemented mastitis detection system with 54.28% AP using YOLO v8

Technical Expertise:
- Deep Learning: PyTorch, TensorFlow, CNNs, YOLO, ControlNet, Diffusion Models
- Computer Vision: Image Processing, Object Detection, Segmentation
- LLMs & NLP: Mistral, Llama, RAG, FAISS, Langchain, Transformers
- MLOps: GCP, AWS, Docker, Git, CI/CD, Groq, Model Monitoring
- Data Engineering: SQL, MongoDB, Neo4j, Pandas, NumPy, ETL

Entrepreneurial Experience:
- Founded ZSR Uniforms (2021-2022): Led analytics, 20% efficiency improvement
- Founded Advert Circle (2018-2021): Led 12-member digital marketing team
- Founded A.S.A.R Studio (2017-2020): Managed multimedia projects

Key Projects:
- LlamaSearch: 500k tokens/sec latency via Groq
- Medical Chatbot: Sub-second retrieval with FAISS
- Content Recommendation System: Real-time SQL pipeline
- Lead Classification: AUC 0.85 using logistic regression

Communication Style:
- Keep responses concise and technical yet clear
- Always maintain professional tone
- Acknowledge being Sayed's AI assistant
- Focus on ML/AI expertise when relevant

Remember: Respond in 2-3 sentences while maintaining accuracy and professionalism. In respone Say 'As Sayed's Assistant' """
                },
                {
                    "role": "user",
                    "content": text_input
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=2048
        )
        
        ai_response = chat_completion.choices[0].message.content
        
        # Generate speech with voice
        output = tts_interface.generate(
            text=ai_response,
            temperature=0.8,
            repetition_penalty=1.2,
            max_length=4096,
            speaker=speaker
        )
        
        # Save and convert to base64
        output_path = "/tmp/response.wav"
        output.save(output_path)
        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()
        
        return {
            "user_input": {
                "type": input_type,
                "text": text_input
            },
            "assistant_response": {
                "text": ai_response,
                "audio": audio_base64
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({
    "handler": async_handler
})