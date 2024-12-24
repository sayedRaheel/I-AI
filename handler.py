import runpod
import asyncio
import base64
import outetts
import torch
from groq import Groq
import os
import time  # Add this import

# Global variables
tts_interface = None
speaker = None
groq_client = None

def initialize_models():
    global tts_interface, speaker, groq_client
    
    if tts_interface is None:
        start_time = time.time()
        print("Starting model initialization...")
        
        model_config = outetts.HFModelConfig_v1(
            model_path="OuteAI/OuteTTS-0.2-500M",
            language="en",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        tts_interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)
        print(f"Model loaded in {time.time() - start_time:.2f}s")
        
        try:
            speaker = tts_interface.load_speaker("sayed_voice.json")
            print("Speaker profile loaded")
        except:
            speaker = tts_interface.load_default_speaker(name="male_1")
            print("Using default speaker")
        
        groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        print(f"Total initialization time: {time.time() - start_time:.2f}s")
    
    return tts_interface, speaker, groq_client

async def async_handler(job):
    try:
        start_time = time.time()
        print("\n=== New Request Started ===")
        
        global tts_interface, speaker, groq_client
        
        if tts_interface is None:
            print("First-time initialization...")
            tts_interface, speaker, groq_client = initialize_models()
        else:
            print("Using existing model")
            
        # Get input from job
        input_type = job["input"]["type"]
        
        if input_type == "text":
            text_input = job["input"]["text"]
            print("Processing text input")
        else:
            print("Processing audio input...")
            audio_start = time.time()
            audio_base64 = job["input"]["audio"]
            audio_bytes = base64.b64decode(audio_base64)
            
            temp_filename = "/tmp/temp_recording.wav"
            with open(temp_filename, "wb") as f:
                f.write(audio_bytes)
                
            with open(temp_filename, "rb") as file:
                translation = groq_client.audio.translations.create(
                    file=(temp_filename, file.read()),
                    model="whisper-large-v3",
                    response_format="json",
                    temperature=0.0
                )
            text_input = translation.text
            print(f"Audio transcription took {time.time() - audio_start:.2f}s")
        
        # LLM Response
        llm_start = time.time()
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": """You are Sayed Raheel Hussain's AI assistant. Provide concise, professional responses based on these facts:

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

Remember: Respond in 2-3 sentences while maintaining accuracy and professionalism. In respone Say 'As Sayed's Assistant' """},
                {"role": "user", "content": text_input}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=2048
        )
        ai_response = chat_completion.choices[0].message.content
        print(f"LLM response took {time.time() - llm_start:.2f}s")
        
        # TTS Generation
        tts_start = time.time()
        with torch.inference_mode():
            print("Starting TTS generation...")
            output = tts_interface.generate(
                text=ai_response,
                temperature=0.8,
                repetition_penalty=1.2,
                max_length=4096,
                speaker=speaker
            )
        print(f"TTS generation took {time.time() - tts_start:.2f}s")
        
        # Save and convert
        output_path = "/tmp/response.wav"
        output.save(output_path)
        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()
            
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        if os.path.exists(output_path):
            os.remove(output_path)
            
        print(f"Total request time: {time.time() - start_time:.2f}s")
        
        return {
            "user_input": {"type": input_type, "text": text_input},
            "assistant_response": {"text": ai_response, "audio": audio_base64}
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"error": str(e)}

print("Starting server...")
initialize_models()
print("Server ready!")

runpod.serverless.start({
    "handler": async_handler
})