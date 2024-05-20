import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials


def load_face_encodings(image_paths):
    encodings = []
    for path in image_paths:
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)[0]
        encodings.append(encoding)
    return encodings


def initialize_csv(file_name):
    f = open(file_name, 'w+', newline='')
    lnwriter = csv.writer(f)
    return f, lnwriter


def authenticate_google_sheets():
    creds_path = 'credentials.json'  # Update with your actual path
    scope = ['https://www.googleapis.com/auth/spreadsheets',
             'https://www.googleapis.com/auth/drive.file']
    creds = Credentials.from_service_account_file(creds_path, scopes=scope)
    client = gspread.authorize(creds)
    return client


def create_or_open_daily_sheet(client, sheet_name, date):
    try:
        sheet = client.open(sheet_name)
    except gspread.SpreadsheetNotFound:
        sheet = client.create(sheet_name)
        # Share the sheet if required, remove or update the email address as needed
        # sheet.share('your-email@example.com', perm_type='user', role='writer')
    try:
        worksheet = sheet.worksheet(date)
    except gspread.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=date, rows="1000", cols="2")
    return worksheet


def log_to_google_sheet(sheet, name, current_time):
    sheet.append_row([name, current_time])


def main():
    video_capture = cv2.VideoCapture(0)

    known_faces_paths = ["faces/Leo Messi.jpg", "faces/Aritra.jpg", "faces/Zlatan.jpg"]
    known_faces_names = ["Leo Messi", "Aritra", "Zlatan"]

    known_face_encodings = load_face_encodings(known_faces_paths)

    students = known_faces_names.copy()

    current_date = datetime.now().strftime("%Y-%m-%d")
    csv_file, csv_writer = initialize_csv(current_date + '.csv')

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    client = authenticate_google_sheets()
    sheet_name = "Attendance Sheet"
    worksheet = create_or_open_daily_sheet(client, sheet_name,
                                           current_date)  # Create or open a sheet with the current date as name

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]

                face_names.append(name)
                if name in known_faces_names and name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H-%M-%S")
                    csv_writer.writerow([name, current_time])
                    log_to_google_sheet(worksheet, name, current_time)  # Log to Google Sheet
                    print(f"{name} marked as present at {current_time}")

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Draw "Detected" label above the face
            cv2.rectangle(frame, (left, top - 35), (right, top), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "Detected", (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    csv_file.close()


if __name__ == "__main__":
    main()






