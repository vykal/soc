import cv2
import face_recognition

# Načítanie videa zo zariadenia so webovou kamerou
video_capture = cv2.VideoCapture(0)

# Načítanie známej tváre a jej kódovanie
known_image = face_recognition.load_image_file("juro.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Inicializácia premenných
face_locations = []
face_encodings = []

while True:
    # Načítanie jedného snímku z videa
    ret, frame = video_capture.read()

    # Konverzia snímku na farebný prenos
    rgb_frame = frame[:, :, ::-1]

    # Detekcia tvárí na snímke
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Cyklus cez všetky detekované tváre
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Porovnanie kódovania detekovanej tváre s kódovaním známej tváre
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        # Zobrazenie výsledku porovnania
        name = "Unknown"
        if matches[0]:
            name = "known"

        # Zobrazenie obdĺžnika okolo detekovanej tváre
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Zobrazenie mena osoby pod obdĺžnikom
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Zobrazenie snímku s detekovanými tvárami na obrazovke
    cv2.imshow('Video', frame)

    # Ak bolo stlačené tlačidlo 'q', ukončiť program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ukončenie videa a zatvorenie okna
video_capture.release()
cv2.destroyAllWindows()
