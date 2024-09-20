import cv2
import os

# Specify the folder that you have already created and the user's name
# Example using an existing folder
folder_name = 'P5_front2'  # Change to the folder you want, e.g., P1_CEO, P2_Salesman, etc.
user_name = 'Yong'  # Change to the user's name as desired, e.g., Ton
save_path = f'./P5_front2'  # The path where the image files will be saved

# Check if the folder exists (do not create a new one if it already exists)
if not os.path.exists(save_path):
    print(f"Folder {save_path} does not exist. Please ensure that the folder is present.")
    exit()

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the camera
cap = cv2.VideoCapture(0)
count = 1  # Frame counter for saving images

while True:
    # Read the image from the camera
    ret, frame = cap.read()
    if not ret:
        print("Cannot access the camera")
        break

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around faces and display frame information
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Add text showing P1, P2, or P3 with the frame count
        text = f"{folder_name}, Frame {count}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Crop the face from the frame
        face_img = frame[y:y + h, x:x + w]

        # Save the cropped face image into the specified folder
        img_name = f"{str(count).zfill(2)}_{user_name}.jpg"
        cv2.imwrite(os.path.join(save_path, img_name), face_img)

        count += 1

    # Show the frame with the drawn rectangles and text
    cv2.imshow('Face Capture', frame)

    # If 'q' is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the display window
cap.release()
cv2.destroyAllWindows()
