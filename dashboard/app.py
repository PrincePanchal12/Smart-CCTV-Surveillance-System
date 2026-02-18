import streamlit as st
import subprocess
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Smart CCTV Surveillance System",
    page_icon="📷",
    layout="centered"
)

st.title("📹 Smart CCTV Surveillance System")
st.markdown("### MCA Final Year Project – AI-Based CCTV Analytics")

st.divider()

# =========================
# Session state
# =========================
if "selected_module" not in st.session_state:
    st.session_state.selected_module = "HOME"

# =========================
# Sidebar navigation
# =========================
st.sidebar.title("📂 Modules")

if st.sidebar.button("👤 Person Detection"):
    st.session_state.selected_module = "PERSON"

if st.sidebar.button("🚨 Intrusion Detection"):
    st.session_state.selected_module = "INTRUSION"

if st.sidebar.button("👥 Crowd Surveillance"):
    st.session_state.selected_module = "CROWD"

# =========================
# HOME
# =========================
if st.session_state.selected_module == "HOME":
    st.subheader("🏠 Dashboard Home")
    st.markdown("""
    **Available Modules**
    - 👤 Person Detection (Live)
    - 🚨 Intrusion Detection (Live + Alerts)
    - 👥 Crowd Surveillance (Live + Trend Analysis)

    👉 Select a module from the sidebar.
    """)

# =========================
# PERSON DETECTION
# =========================
elif st.session_state.selected_module == "PERSON":
    st.subheader("👤 Person Detection")

    if st.button("▶ Start Person Detection"):
        subprocess.Popen(
            "yolo detect predict "
            "model=runs/detect/smart_cctv_v13/weights/best.pt "
            "source=0 device=0 conf=0.4 show=True",
            shell=True
        )
        st.success("Person Detection Started (Webcam Opened)")

# =========================
# INTRUSION DETECTION
# =========================
elif st.session_state.selected_module == "INTRUSION":
    st.subheader("🚨 Intrusion Detection")

    if st.button("▶ Start Intrusion Detection"):
        subprocess.Popen(
            f"{sys.executable} intrusion_detection/run_intrusion_detection.py",
            shell=True
        )
        st.success("Intrusion Detection Started (Webcam Opened)")

# =========================
# CROWD SURVEILLANCE
# =========================
elif st.session_state.selected_module == "CROWD":
    st.subheader("👥 Crowd Surveillance")

    st.markdown("""
    - Start live crowd monitoring (webcam)
    - View crowd trend graph from logged data
    """)

    # ---------- Start Crowd Surveillance ----------
    if st.button("▶ Start Crowd Surveillance (Webcam)"):
        subprocess.Popen(
            f"{sys.executable} person_detection/crowd_surveillance.py",
            shell=True
        )
        st.success("Crowd Surveillance Started (Webcam Opened)")

    st.divider()

    # ---------- Trend Analysis ----------
    st.subheader("📈 Crowd Trend Analysis")

    log_file = "person_detection/crowd_log.csv"

    if not os.path.exists(log_file):
        st.warning("Crowd log file not found. Start crowd surveillance first.")
    else:
        try:
            df = pd.read_csv(log_file)

            if df.empty:
                st.info("Crowd log file exists but no data recorded yet. Please wait 1–2 minutes.")
            else:
                df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])

                fig, ax = plt.subplots()
                ax.plot(df["Datetime"], df["People_Count"], marker="o")
                ax.set_xlabel("Time")
                ax.set_ylabel("People Count")
                ax.set_title("Crowd Trend Over Time")
                ax.grid(True)

                st.pyplot(fig)

        except EmptyDataError:
            st.info("Crowd log file exists but contains no data yet.")

st.divider()
st.caption("© Smart CCTV – MCA Final Year Project")
