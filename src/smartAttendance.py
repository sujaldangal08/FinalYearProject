import cv2
import os
import numpy as np
import datetime
import time
import pandas as pd

# Function to load images and labels from a directory
def load_images_from_folder(folder):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for subdir in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, subdir)):
            label_dict[subdir] = current_label
            current_label += 1

            for filename in os.listdir(os.path.join(folder, subdir)):
                img_path = os.path.join(folder, subdir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label_dict[subdir])

    return images, labels, label_dict

# Load training images, labels, and label-to-name mapping
training_folder = "/home/sujaldangal/Documents/training_folder"
training_images, training_labels, label_dict = load_images_from_folder(training_folder)

# Function to calculate LBPH feature for an image
def calculate_lbp_image(img):
    height, width = img.shape
    lbp_image = np.zeros((height-2, width-2), dtype=np.uint8)

    for y in range(1, height-1):
      for x in range(1, width-1):
        center = img[y, x]
        pattern = np.uint8(0)  # Initialize pattern as uint8

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if img[y + dy, x + dx] >= center:
                    pattern |= np.uint8(1)  # Set the least significant bit
                pattern <<= np.uint8(1)  # Left-shift the pattern

        # Right-shift the pattern to keep it within 8 bits
        pattern >>= np.uint8(1)

        lbp_image[y - 1, x - 1] = pattern


    return lbp_image


# Function to calculate LBPH histogram for an image
def calculate_lbp_histogram(lbp_image, num_bins=256):
    hist, _ = np.histogram(lbp_image.ravel(), bins=num_bins, range=(0, num_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram

    return hist

# Calculate LBPH histograms for training images
lbph_histograms = []
for img in training_images:
    lbp_image = calculate_lbp_image(img)
    lbp_histogram = calculate_lbp_histogram(lbp_image)
    lbph_histograms.append(lbp_histogram)

# Initialize the video capture for webcam
video_capture = cv2.VideoCapture(0)

# Attendance records
attendance = {}

# Cooldown period in seconds
cooldown_period = 300

# Create a DataFrame to store attendance data
attendance_df = pd.DataFrame(columns=['Name', 'Timestamp'])

while True:
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = gray_frame[y:y + h, x:x + w]

        # Calculate LBPH feature for the detected face
        lbp_image = calculate_lbp_image(face)
        lbp_histogram = calculate_lbp_histogram(lbp_image)

        # Match the LBPH histogram with training histograms
        min_distance = float("inf")
        recognized_label = -1

        for i, hist in enumerate(lbph_histograms):
            distance = np.linalg.norm(hist - lbp_histogram)
            if distance < min_distance:
                min_distance = distance
                recognized_label = training_labels[i]

        # Get the recognized person's name
        recognized_person = "Unknown"
        for name, idx in label_dict.items():
            if idx == recognized_label:
                recognized_person = name
                break

        # Record attendance
        if recognized_person != "Unknown":
            if recognized_person not in attendance:
                attendance[recognized_person] = []

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not attendance[recognized_person] or (time.time() - attendance[recognized_person][-1]) >= cooldown_period:
                attendance[recognized_person].append(time.time())  # Record the current time
                
                # Add data to the DataFrame
                attendance_df = pd.concat([attendance_df, pd.DataFrame({'Name': [recognized_person], 'Timestamp': [current_time]})])

        # Draw a rectangle and label for the recognized face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, recognized_person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()

# Save attendance records to an Excel file
excel_file_path = "/home/sujaldangal/Documents/attendance.xlsx"

try:
    attendance_df.to_excel(excel_file_path, index=False)
    print(f"Attendance data saved to {excel_file_path}")
except Exception as e:
    print(f"An error occurred while saving the Excel file: {e}")
