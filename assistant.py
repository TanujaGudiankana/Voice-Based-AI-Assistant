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
import cv2
import face_recognition
import numpy as np
from pathlib import Path

# Global variables for face recognition
KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []
face_recognition_enabled = False

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
    "train_face": ["train face", "learn face", "remember face", "add face"],
    "exit": ["exit", "stop", "quit", "goodbye", "bye"]
}

def initialize_face_recognition():
    """Initialize face recognition by loading known faces"""
    global known_face_encodings, known_face_names, face_recognition_enabled
    
    # Create known_faces directory if it doesn't exist
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        return
    
    # Load known faces
    for image_file in os.listdir(KNOWN_FACES_DIR):
        if image_file.endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(image_file)[0]
            image_path = os.path.join(KNOWN_FACES_DIR, image_file)
            
            # Load and encode face
            face_image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(face_image)
            
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(name)
    
    face_recognition_enabled = len(known_face_encodings) > 0
    return face_recognition_enabled

def train_new_face(name):
    """Train the system to recognize a new face"""
    global known_face_encodings, known_face_names, face_recognition_enabled
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    speak(f"Please look at the camera for {name}'s face training")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Display the frame
        cv2.imshow('Training Face', frame)
        
        # Check for face
        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            # Save the image
            image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            cv2.imwrite(image_path, frame)
            
            # Add to known faces
            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
            
            face_recognition_enabled = True
            speak(f"Successfully trained face for {name}")
            break
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def recognize_face():
    """Recognize faces in real-time"""
    if not face_recognition_enabled:
        return None
    
    cap = cv2.VideoCapture(0)
    recognized_name = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Find faces in frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                recognized_name = known_face_names[first_match_index]
                break
        
        if recognized_name:
            break
        
        # Break loop after 5 seconds if no face recognized
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return recognized_name

def get_best_command_match(user_input):
    """Find the best matching command template"""
    best_match = None
    highest_ratio = 0
    
    for command_type, templates in COMMAND_TEMPLATES.items():
        for template in templates:
            ratio = difflib.SequenceMatcher(None, user_input, template).ratio()
            if ratio > highest_ratio and ratio > 0.5:  # Lower threshold for better matching
                highest_ratio = ratio
                best_match = command_type
    
    return best_match

def speak(text):
    try:
        # Initialize pyttsx3 engine
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        
        # Set female voice (usually index 1)
        female_voice = None
        for voice in voices:
            if "female" in voice.name.lower():
                female_voice = voice.id
                break
        
        if female_voice:
            engine.setProperty('voice', female_voice)
        else:
            # If no female voice found, try to use the second voice (usually female)
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
        
        # Set properties for better voice quality
        engine.setProperty('rate', 150)    # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in speech synthesis: {str(e)}")
        # If pyttsx3 fails, just print the text
        print(f"Text to speak: {text}")

def listen():
    recognizer = sr.Recognizer()
    text = ""
    error_count = 0
    max_retries = 3

    while not text and error_count < max_retries:
        try:
            with sr.Microphone() as source:
                print("Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("Listening...")
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    print("Processing speech...")
                    # Try Google recognition
                    try:
                        text = recognizer.recognize_google(audio).lower()
                        print(f"Recognized: {text}")
                    except:
                        error_count += 1
                        if error_count < max_retries:
                            speak("I didn't catch that. Please try again.")
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
            # If PyAudio is not found, provide a helpful message
            if "PyAudio" in str(e):
                speak("PyAudio is not installed. Please install it using: pip install pyaudio")
                return ""

    return text if text else ""

def open_file(file_name):
    try:
        file_path = os.path.join(os.path.expanduser("~"), "Documents", f"{file_name}.txt")
        if os.path.exists(file_path):
            os.startfile(file_path)
            return f"Opening {file_name}"
        return "File not found."
    except Exception as e:
        print(f"Error opening file: {str(e)}")
        return f"Error opening file: {str(e)}"

def search_file(file_name, keyword):
    try:
        file_path = os.path.join(os.path.expanduser("~"), "Documents", f"{file_name}.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().lower()
                # Use more sophisticated search
                words = content.split()
                keyword_parts = keyword.lower().split()
                matches = []
                
                for i in range(len(words)):
                    if all(kw in ' '.join(words[i:i+len(keyword_parts)]).lower() for kw in keyword_parts):
                        matches.append(' '.join(words[i:i+len(keyword_parts)]))
                
                if matches:
                    return f"Found matches: {', '.join(matches[:3])}"
        return f"No matches found for '{keyword}'"
    except Exception as e:
        print(f"Error searching file: {str(e)}")
        return f"Error searching file: {str(e)}"

def google_search(query):
    try:
        url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for g in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
            results.append(g.text)
        return results[:3] if results else ["No results found"]
    except Exception as e:
        print(f"Error in Google search: {str(e)}")
        return [f"Error performing search: {str(e)}"]

def youtube_search(query):
    try:
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
        return f"Searching YouTube for {query}"
    except Exception as e:
        print(f"Error in YouTube search: {str(e)}")
        return f"Error searching YouTube: {str(e)}"

def process_command(command):
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
                        # Use more sophisticated search
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
        
        elif best_match == "train_face":
            speak("What is the name of the person?")
            name = listen()
            if name:
                if train_new_face(name):
                    return f"Successfully trained face for {name}"
                return "Failed to train face"
            return "I couldn't understand the name."
        
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
            results = google_search(query)
            return "Here are the top results: " + " ".join(results)
        
        elif best_match == "youtube_search":
            query = command.replace("search youtube for", "").strip()
            if not query:
                speak("What would you like to search for on YouTube?")
                query = listen()
                if not query:
                    return "I couldn't understand your search query."
            
            return youtube_search(query)
        
        elif best_match == "time":
            return f"The time is {datetime.datetime.now().strftime('%H:%M')}"
        
        elif best_match == "exit":
            return "Goodbye!"
        
        return "I didn't understand. Please try again."
    
    except Exception as e:
        print(f"Error processing command: {str(e)}")
        return f"An error occurred: {str(e)}"

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

def main():
    # Check if PyAudio is installed
    try:
        import pyaudio
        print("PyAudio is installed.")
    except ImportError:
        print("PyAudio is not installed. Please install it using: pip install pyaudio")
        print("On Windows, you may need to install it using: pip install pipwin")
        print("Then: pipwin install pyaudio")
    
    # Initialize face recognition
    if initialize_face_recognition():
        speak("Face recognition system initialized")
    
    speak("Hello! My name is Friday, I am your assistant.\n"
          "Here are my functionalities:")
    functionalities = [
        "Open files",
        "Search in files",
        "Open calculator",
        "Open notepad",
        "Open Chrome",
        "Search Google",
        "Search YouTube",
        "Check time",
        "Train faces",
        "Recognize people"
    ]
    
    for func in functionalities:
        speak(func)
    
    speak("What would you like me to do?")
    
    # Start continuous listening in a separate thread
    listen_thread = threading.Thread(target=continuous_listen)
    listen_thread.daemon = True
    listen_thread.start()
    
    # Keep the main thread alive
    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    main()
