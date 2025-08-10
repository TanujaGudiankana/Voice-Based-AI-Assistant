import cv2
import os
import time
import face_recognition

def test_camera():
    """Test if camera is working"""
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    print("Camera opened successfully")
    print("Attempting to capture frame...")
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from camera")
        cap.release()
        return False
    
    print("Successfully captured frame")
    
    # Save the test image
    if not os.path.exists("test_images"):
        os.makedirs("test_images")
    
    test_image_path = "test_images/camera_test.jpg"
    cv2.imwrite(test_image_path, frame)
    print(f"Saved test image to {test_image_path}")
    
    # Show the image for a few seconds
    cv2.imshow('Camera Test', frame)
    cv2.waitKey(3000)  # Wait for 3 seconds
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    print("Camera test: Success")
    return True

def test_face_detection():
    """Test if face detection is working with live preview"""
    print("\nStarting face detection test...")
    print("This will show a live preview with face detection.")
    print("Press 'q' to quit or 's' to save when your face is detected.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    face_found = False
    saved_image = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(frame)
        
        # Draw rectangles around faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            face_found = True
        
        # Display the resulting frame
        cv2.imshow('Face Detection Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and face_found:
            # Save the image with face
            if not os.path.exists("known_faces"):
                os.makedirs("known_faces")
            cv2.imwrite("known_faces/test_face.jpg", frame)
            print("\nSaved face image to known_faces/test_face.jpg")
            saved_image = True
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    if face_found:
        print("Face detection test: Success")
        if not saved_image:
            print("Note: No image was saved. You can run the test again and press 's' to save an image.")
        return True
    else:
        print("Face detection test: No faces found")
        return False

def test_face_recognition():
    """Test if face recognition is working with a saved face"""
    print("\nStarting face recognition test...")
    print("This will compare your face with a previously saved face.")
    print("Press 'q' to quit.")
    
    # Check if we have a saved face to compare against
    known_face_path = "known_faces/test_face.jpg"
    if not os.path.exists(known_face_path):
        print("Error: No known face image found. Please run the face detection test first and save an image.")
        return False
    
    # Load the known face
    known_image = face_recognition.load_image_file(known_face_path)
    known_face_encodings = face_recognition.face_encodings(known_image)
    
    if not known_face_encodings:
        print("Error: No face found in the known face image.")
        return False
    
    known_face_encoding = known_face_encodings[0]
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Find faces in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        # Process each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known face
            matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
            name = "Known Person" if matches[0] else "Unknown"
            
            # Draw rectangle and label
            color = (0, 255, 0) if matches[0] else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display the resulting frame
        cv2.imshow('Face Recognition Test', frame)
        
        # Handle key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    print("Face recognition test completed")
    return True

def main():
    print("Starting face recognition tests...")
    
    # Test 1: Camera
    print("\nTest 1: Testing camera...")
    print("Press Enter to start camera test...")
    input()
    
    if not test_camera():
        return
    
    # Test 2: Face Detection
    print("\nTest 2: Testing face detection...")
    print("Press Enter to start face detection test...")
    input()
    
    if not test_face_detection():
        return
    
    # Test 3: Face Recognition
    print("\nTest 3: Testing face recognition...")
    print("Press Enter to start face recognition test...")
    input()
    
    if not test_face_recognition():
        return
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
