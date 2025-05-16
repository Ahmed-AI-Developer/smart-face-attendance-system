import streamlit as st
import cv2
import os
import time
import pandas as pd
from deepface import DeepFace
from utils import mark_attendance

st.set_page_config(page_title="Smart Attendance", layout="wide")

# --- Ensure directories ---
os.makedirs("database/faces", exist_ok=True)

# --- Sidebar ---
st.sidebar.title("📊 Control Panel")
choice = st.sidebar.selectbox("Select Action", ["🏠 Home", "➕ Add User", "🧑‍🤝‍🧑 Mark Attendance", "📁 Download CSV", "👤 Registered Users"])

# --- Main Title ---
st.title("📸 Smart Attendance System")

# --- Home Page ---
if choice == "🏠 Home":
    st.image("assets_images/banner.png", use_container_width=True)
    st.markdown("""
    ## 👋 Welcome!
    This is an AI-based Smart Attendance System using **DeepFace**.
    
    ### Features:
    - ➕ Add new users via webcam
    - 🧑‍🤝‍🧑 Mark attendance in real-time
    - 📁 Download attendance reports
    - 👤 View registered users
    """)
    st.info("Use the sidebar to navigate.")

# --- Add User ---
elif choice == "➕ Add User":
    st.subheader("Add a New User")
    name = st.text_input("Enter full name")

    if st.button("📸 Capture Face"):
        if name.strip() == "":
            st.warning("Please enter a name first.")
        else:
            cap = cv2.VideoCapture(0)
            st.info("Press 's' to capture, 'q' to quit.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Webcam not working.")
                    break

                cv2.imshow("Capture Face - Press 's'", frame)
                key = cv2.waitKey(1)

                if key == ord('s'):
                    filename = f"database/faces/{name.strip().lower()}.jpg"
                    cv2.imwrite(filename, frame)
                    st.success(f"✅ Face saved for {name}")
                    break
                elif key == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

# --- Mark Attendance ---
elif choice == "🧑‍🤝‍🧑 Mark Attendance":
    st.subheader("Mark Attendance with Webcam")
    if st.button("Start Attendance"):
        known_faces = []
        names = []

        for file in os.listdir("database/faces"):
            if file.endswith((".jpg", ".png")):
                known_faces.append(f"database/faces/{file}")
                names.append(os.path.splitext(file)[0])

        cap = cv2.VideoCapture(0)
        marked = []
        frame_count = 0
        skip_frames = 10

        st.info("Webcam started. Press 'q' to stop window.")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam error.")
                break

            small_frame = cv2.resize(frame, (320, 240))
            frame_count += 1

            if frame_count % skip_frames == 0:
                for i, face_path in enumerate(known_faces):
                    result = DeepFace.verify(small_frame, face_path, enforce_detection=False)
                    if result['verified']:
                        person = names[i]
                        if person not in marked:
                            mark_attendance(person)
                            marked.append(person)
                            st.success(f"✅ Marked: {person}")
                            time.sleep(1)

            cv2.imshow("Marking Attendance - Press 'q' to exit", frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# --- Download CSV ---
elif choice == "📁 Download CSV":
    st.subheader("Download Attendance Record")
    try:
        df = pd.read_csv("database/attendence.csv", names=["Name", "Timestamp"])
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "attendance.csv", "text/csv")
    except FileNotFoundError:
        st.warning("Attendance file not found.")

# --- Show Registered Users ---
elif choice == "👤 Registered Users":
    st.subheader("Registered Users")
    users = [f.split('.')[0] for f in os.listdir("database/faces") if f.endswith((".jpg", ".png"))]
    if users:
        st.write("**Total Users:**", len(users))
        st.write(users)
    else:
        st.info("No registered users found.")
