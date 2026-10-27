from dotenv import load_dotenv
load_dotenv()
import asyncio
import io
import re
import os
import pygame
import edge_tts
import speech_recognition as sr
from langchain_core.messages import AIMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from .graph import create_chat_graph



mongodb_uri = "mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "user-session-1"}}

# Initialize audio mixer
pygame.mixer.init()

def clean_text_for_tts(text: str) -> str:
    """Strips Markdown syntax, code blocks, and special characters before speaking."""
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove inline code
    text = re.sub(r'`.*?`', '', text)
    # Remove markdown bold/italic/links
    text = re.sub(r'[*_~#]', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    return text.strip()

async def _edge_speak_async(text: str, voice: str = "en-IN-PrabhatNeural"):
    """
    High-quality Neural Voices options:
    - Hindi/Indian English: "en-IN-PrabhatNeural", "en-IN-NeerjaNeural", "hi-IN-MadhurNeural"
    - US English: "en-US-AndrewNeural", "en-US-AriaNeural"
    """
    clean_text = clean_text_for_tts(text)
    if not clean_text:
        return

    communicate = edge_tts.Communicate(clean_text, voice=voice)
    mp3_data = bytearray()
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data.extend(chunk["data"])

    # Play in-memory MP3 without writing temp files to disk
    audio_stream = io.BytesIO(mp3_data)
    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def speak(text: str):
    """Sync wrapper to execute edge-tts."""
    try:
        asyncio.run(_edge_speak_async(text))
    except Exception as e:
        print(f"TTS Playback Error: {e}")

def main():
    with MongoDBSaver.from_conn_string(mongodb_uri) as checkpointer:
        r = sr.Recognizer()
        graph = create_chat_graph(checkpointer=checkpointer)

        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1.0)
            r.pause_threshold = 0.8

            while True:
                print("\nListening...")
                try:
                    audio = r.listen(source)
                    print("Processing speech...")
                    sst = r.recognize_google(audio)
                    print(f"You said: {sst}")

                    last_ai_text = None

                    # Stream graph responses
                    for event in graph.stream(
                        {"messages": [{"role": "user", "content": sst}]},
                        config,
                        stream_mode="values",
                    ):
                        if "messages" in event:
                            msg = event["messages"][-1]
                            msg.pretty_print()

                            if isinstance(msg, AIMessage) and msg.content:
                                if isinstance(msg.content, str):
                                    last_ai_text = msg.content
                                elif isinstance(msg.content, list):
                                    texts = [
                                        item.get("text", "")
                                        for item in msg.content
                                        if isinstance(item, dict) and "text" in item
                                    ]
                                    last_ai_text = " ".join(texts)

                    # Speak out the AI's response automatically
                    if last_ai_text:
                        speak(last_ai_text)

                except sr.UnknownValueError:
                    print("Could not understand audio.")
                except KeyboardInterrupt:
                    print("\nStopping...")
                    break
                except Exception as e:
                    print(f"Error: {e}")


if __name__ == "__main__":
    main()