import os
import io
import base64
import keyboard
import mss
from PIL import Image
from groq import Groq

# Initialize the Groq client
# It will automatically look for the GROQ_API_KEY environment variable
client = Groq()

def capture_and_solve():
    print("\n[+] Capturing screen...")
    
    # 1. Ultra-fast screen capture
    with mss.MSS() as sct:
        # Grab the primary monitor
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        
        # Convert raw pixels to a PIL Image
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    
    # 2. Resize and compress to minimize network latency
    img.thumbnail((1600, 1600)) 
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 3. Call the Llama 3.2 Vision API
    prompt = "Look at this Kahoot question and options. Read the text, solve the trivia, and reply with ONLY the correct color (Red, Blue, Yellow, or Green). Note: For True/False questions, True is Red and False is Blue. Give zero explanation."
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="qwen/qwen3.8-27b",
            temperature=0.0, # Zero creativity, maximum accuracy
            max_tokens=10,   # Cut off generation immediately after the answer
        )
        answer = chat_completion.choices[0].message.content.strip()
        print(f"\n============================")
        print(f" ANSWER: {answer.upper()}")
        print(f"============================\n")
    except Exception as e:
        print(f"Error: {e}")

print("Kahoot Sniper is running.")
print("Press 'Ctrl + Shift + Space' to trigger.")
print("Press 'Esc' to exit.")

# Bind to your hotkey of choice
keyboard.add_hotkey('ctrl+shift+space', capture_and_solve)

# Keep the script listening in the background
keyboard.wait('esc')