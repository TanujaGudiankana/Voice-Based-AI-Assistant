from flask import Flask, render_template, request, jsonify
import os
import pyttsx3
import speech_recognition as sr
import webbrowser
import datetime
import subprocess
import playsound
from gtts import gTTS
import requests
from bs4 import BeautifulSoup
import difflib
import json
import time
import threading
import tempfile
import cv2
import face_recognition
import numpy as np
from PIL import Image
import random

# Initialize face recognition variables
known_faces = {}  # Dictionary to store known faces and their names
current_user = None  # Currently recognized user

# Command templates for better matching
COMMAND_TEMPLATES = {
    "open_file": ["open file", "open the file", "open document", "open a file"],
    "search_file": ["search in file", "search file", "find in file", "search for in file"],
    "calculator": ["open calculator", "launch calculator", "start calculator", "calculator"],
    "notepad": ["open notepad", "launch notepad", "start notepad", "notepad"],
    "chrome": ["open chrome", "launch chrome", "start chrome", "chrome"],
    "google_search": ["search google for", "google search", "search for", "search", "find"],
    "youtube_search": ["search youtube for", "youtube search", "find on youtube", "youtube"],
    "time": ["what is the time", "current time", "time now", "tell me the time"],
    "exit": ["exit", "stop", "quit", "goodbye", "bye"]
}

def speak(text):
    """Speak the given text using pyttsx3 with female voice"""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        # Try to find a female voice
        female_voice = None
        for voice in voices:
            if "female" in voice.name.lower():
                female_voice = voice.id
                break
        
        if female_voice:
            engine.setProperty('voice', female_voice)
        elif len(voices) > 1:  # If no female voice found, use the second voice (usually female)
            engine.setProperty('voice', voices[1].id)
        
        # Set properties for better voice quality
        engine.setProperty('rate', 150)    # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in speech synthesis: {str(e)}")
        # Fallback to gTTS if pyttsx3 fails
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts = gTTS(text=text, lang='en')
                tts.save(fp.name)
                playsound.playsound(fp.name)
                os.unlink(fp.name)
        except Exception as e2:
            print(f"Error in fallback speech synthesis: {str(e2)}")

def listen():
    """Listen for voice input with improved error handling"""
    recognizer = sr.Recognizer()
    text = ""
    error_count = 0
    max_retries = 3

    while not text and error_count < max_retries:
        try:
            with sr.Microphone() as source:
                print("Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1.0)  # Increased from 0.5 to 1.0
                print("Listening...")
                try:
                    audio = recognizer.listen(source, timeout=15, phrase_time_limit=20)  # Increased timeouts
                    print("Processing speech...")
                    try:
                        text = recognizer.recognize_google(audio).lower()
                        print(f"Recognized: {text}")
                    except sr.UnknownValueError:
                        error_count += 1
                        if error_count < max_retries:
                            speak("I didn't catch that. Please try again.")
                        continue
                    except sr.RequestError as e:
                        print(f"Could not request results; {e}")
                        error_count += 1
                        if error_count < max_retries:
                            speak("There was an error with the speech recognition service.")
                        continue
                except sr.WaitTimeoutError:
                    error_count += 1
                    if error_count < max_retries:
                        speak("Listening timeout. Please try again.")
        except Exception as e:
            error_count += 1
            print(f"Error in speech recognition: {str(e)}")
            if error_count < max_retries:
                speak("There was an error. Please try again.")
            if "PyAudio" in str(e):
                speak("PyAudio is not installed. Please install it using: pip install pyaudio")
                return ""

    return text if text else ""

def get_best_command_match(user_input):
    """Find the best matching command template with improved matching"""
    best_match = None
    highest_ratio = 0
    
    for command_type, templates in COMMAND_TEMPLATES.items():
        for template in templates:
            # Try exact match first
            if template in user_input:
                return command_type
            
            # If no exact match, try fuzzy matching
            ratio = difflib.SequenceMatcher(None, user_input, template).ratio()
            if ratio > highest_ratio and ratio > 0.6:  # Increased threshold for better accuracy
                highest_ratio = ratio
                best_match = command_type
    
    return best_match

def process_command(command):
    """Process voice commands with improved error handling"""
    if not command:
        return "I didn't hear anything. Please try again."

    best_match = get_best_command_match(command)
    
    try:
        if best_match == "open_file":
            speak("Please say the file name.")
            file_name = listen()
            if file_name:
                file_path = os.path.join(os.path.expanduser("~"), "Documents", f"{file_name}.txt")
                if os.path.exists(file_path):
                    os.startfile(file_path)
                    return f"Opening {file_name}"
                return "File not found."
            return "I couldn't understand the file name."
        
        elif best_match == "search_file":
            speak("Say the file name.")
            file_name = listen()
            if file_name:
                file_path = os.path.join(os.path.expanduser("~"), "Documents", f"{file_name}.txt")
                if not os.path.exists(file_path):
                    return "File not found."
                
                speak("Say the word to search.")
                keyword = listen()
                if keyword:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read().lower()
                        words = content.split()
                        keyword_parts = keyword.lower().split()
                        matches = []
                        
                        for i in range(len(words)):
                            if all(kw in ' '.join(words[i:i+len(keyword_parts)]).lower() for kw in keyword_parts):
                                matches.append(' '.join(words[i:i+len(keyword_parts)]))
                        
                        if matches:
                            return f"Found matches: {', '.join(matches[:3])}"
                    return f"No matches found for '{keyword}'"
                return "I couldn't understand the search keyword."
            return "I couldn't understand the file name."
        
        elif best_match == "calculator":
            subprocess.Popen("calc")
            return "Opening Calculator."
        
        elif best_match == "notepad":
            subprocess.Popen("notepad")
            return "Opening Notepad."
        
        elif best_match == "chrome":
            subprocess.Popen("start chrome", shell=True)
            return "Opening Google Chrome."
        
        elif best_match == "google_search":
            query = command.replace("search google for", "").strip()
            if not query:
                speak("What would you like to search for?")
                query = listen()
                if not query:
                    return "I couldn't understand your search query."
            
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"Searching Google for {query}"
        
        elif best_match == "youtube_search":
            query = command.replace("search youtube for", "").strip()
            if not query:
                speak("What would you like to search for on YouTube?")
                query = listen()
                if not query:
                    return "I couldn't understand your search query."
            
            webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
            return f"Searching YouTube for {query}"
        
        elif best_match == "time":
            return f"The time is {datetime.datetime.now().strftime('%H:%M')}"
        
        elif best_match == "exit":
            return "Goodbye!"
        
        return "I didn't understand. Please try again."
    
    except Exception as e:
        print(f"Error processing command: {str(e)}")
        return f"An error occurred: {str(e)}"

def initialize_face_recognition():
    """Initialize face recognition by capturing and encoding the user's face"""
    global known_faces, current_user
    
    print("Initializing face recognition...")
    speak("Let me take a look at you to recognize you.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        speak("I'm having trouble accessing the camera. Please make sure it's connected and try again.")
        return False
    
    face_found = False
    frame_count = 0
    max_attempts = 30  # Try for 30 frames before giving up
    
    while not face_found and frame_count < max_attempts:
        ret, frame = cap.read()
        if not ret:
            frame_count += 1
            continue
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            # Get face encoding
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            if face_encodings:
                # Check if this face matches any known faces
                for name, known_encoding in known_faces.items():
                    matches = face_recognition.compare_faces([known_encoding], face_encodings[0])
                    if matches[0]:
                        current_user = name
                        speak(f"Welcome back, {name}!")
                        face_found = True
                        break
                
                if not face_found:
                    # New face detected
                    speak("I don't recognize you. What's your name?")
                    name = listen()
                    if name:
                        name = name.capitalize()
                        # Save the face encoding and name
                        known_faces[name] = face_encodings[0]
                        current_user = name
                        
                        # Save the face image
                        if not os.path.exists("known_faces"):
                            os.makedirs("known_faces")
                        cv2.imwrite(f"known_faces/{name}.jpg", frame)
                        
                        speak(f"Nice to meet you, {name}! I'll remember your face.")
                        face_found = True
                    else:
                        speak("I'll call you User then!")
                        name = "User"
                        known_faces[name] = face_encodings[0]
                        current_user = name
                        face_found = True
        
        frame_count += 1
        time.sleep(0.1)  # Add small delay to prevent CPU overuse
    
    cap.release()
    return face_found

def continuous_listen():
    """Continuously listen for commands"""
    while True:
        command = listen()
        if command:
            response = process_command(command)
            speak(response)
            if response == "Goodbye!":
                break
        time.sleep(0.1)  # Small delay to prevent CPU overuse

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        command = data.get("command", "").lower()
        if command:
            response = process_command(command)
            return jsonify({"response": response, "success": True})
        return jsonify({"response": "No command received", "success": False})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}", "success": False})

if __name__ == '__main__':
    # Check if PyAudio is installed
    try:
        import pyaudio
        print("PyAudio is installed.")
    except ImportError:
        print("PyAudio is not installed. Please install it using: pip install pyaudio")
        print("On Windows, you may need to install it using: pip install pipwin")
        print("Then: pipwin install pyaudio")
    
    # First initialize face recognition
    if initialize_face_recognition():
        # After recognizing the user, introduce the assistant
        time.sleep(1)  # Small pause for natural conversation
        speak(f"Hello {current_user}! I am Friday, your personal AI assistant. "
              "I can help you with various tasks like opening applications, "
              "searching the web, and much more. How can I assist you today?")
        
        # Start continuous listening in a separate thread
        listen_thread = threading.Thread(target=continuous_listen)
        listen_thread.daemon = True
        listen_thread.start()
        
        # Start the Flask app
        app.run(debug=True)
    else:
        speak("I'm having trouble with face recognition. Please make sure your camera is working properly.")
