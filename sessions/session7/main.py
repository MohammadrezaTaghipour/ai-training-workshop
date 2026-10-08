# import streamlit as st

#region Part 1 text,..
# my_text = "MohammadReza Taghipour"
# st.title(my_text)
#
# st.title("Welcome to Stramlit App", help="Greeting")
#
# st.title(":red[Sematec]",help="Sematec")
#
# st.header("Sematec")
# st.subheader("Welcome")
#
#
# sample_text = "<h1>MohammadReza Taghipour</h1>"
# st.markdown(sample_text, unsafe_allow_html=True)
#
# st.markdown("*Welcome*")
# st.markdown("**Welcome**")
# st.markdown("***Welcome***")
#
#
# st.caption(":red[Sematec] :red-background[IT Institude]:",unsafe_allow_html=True)
#
#
# python_Code = '''
# import pandas as pd
# df = pd.read_csv("titanic.csv")")
# '''
# st.code(python_Code)
#
# st.divider()
#
# st.latex(r'''
# \begin{rcases}
#    a &\text{if } b \\
#    c &\text{if } d
# \end{rcases}⇒…
# ''')
#
# st.divider()
#
# st.success("Sematec IT Learning Institude")
# st.warning("Sematec IT Learning Institude")
# st.info("Sematec IT Learning Institude")
# st.error("Sematec IT Learning Institude"
#endregion


#region Part 2 button

# caption:str = "Submit!!!"
# st.button(caption)
# st.button(caption, key="Submit")
# st.button(caption, key="Submit 2", type="primary")
# st.button(caption, key="Submit 3", use_container_width=True)

# st.download_button("Download File!!!",
#                    data="ID, Filename, Lastname, Education",
#                    mime="text/csv",
#                    file_name='data.csv')

# st.link_button("وب سایت گیت", "https://github.com/MohammadrezaTaghipour")
#
# st.divider()
#
# eduction_dic = {
#     1: "Diploma",
#     2: "Bachelor",
#     3: "Master",
#     4: "Phd",
# }
#
# selected = []
# for key, value in eduction_dic.items():
#     checked = st.checkbox(f"{key}: {value}", key=f"edu_{key}")
#     if checked:
#         selected.append(value)
#
# st.write("selected items", selected)

# st.divider()
#
# st.multiselect("List", options=["Diploma", "Bachelor"])
#
# st.divider()
#
# st.toggle("is Active")
#
# st.divider()
#
# st.radio("education", options=["Education", "Masters", "Faculty"], index=2)

# st.divider()
#
# st.selectbox("list",
#              options=["Education", "Masters", "Faculty"],
#              index=1,
#              placeholder="Select your option",
#              disabled=True)
#
# st.time_input("Time", value="now")
# st.date_input("Date", value="today", format="YYYY/MM/DD")
# st.divider()
#
# st.text_input("Text", value="For example: master")
# st.text_input("Password", type="password", max_chars=24)
#
# st.text_area("Text", value="For example: master")
# st.text_area("Descriptipn", max_chars=250, height=300)
#


#region Part 3 Text to speech FirstSample

# uv add --group web edge-tts  # Text to Speech

# import edge_tts as tts
#
#
# voice_male:str = "fa-IR-FaridNeural"
# voice_female:str = "fa-IR-DilaraNeural"
#
# # text: str = "سَلام. دَورِهٔ جامِعِ هُوشِ مَصنُوعی دَر سَماتِک، دَر حالِ بَرگُزاری اَست."
# text: str = "Hi, The AI workshop in Sematech is running"
#
# audio_file_path:str = f"voice_result/{voice_male}.mp3"
#
# communicate = tts.Communicate(
#     text=text,
#     voice=voice_male
# )
#
# communicate.save_sync(
#     audio_fname=audio_file_path,
# )

#endregion


#region Part 4 Test to Speech - From File
# from pathlib import Path
# from edge_tts import Communicate

# voice: str = "fa-IR-DilaraNeural"
#
# session_dir = Path(__file__).resolve().parent
# sample_path = session_dir / "sample_file.txt"
# out_dir = session_dir / "voice_result"
# out_dir.mkdir(exist_ok=True)
# audio_file_path = str(out_dir / f"{voice}.mp3")
#
# with open(sample_path, "r", encoding="utf-8") as file:
#     text: str = file.read()
#
# communicate = Communicate(
#     text=text,
#     voice=voice,
#     rate="+5%",
#     pitch="+10Hz",
# )
#
# communicate.save_sync(audio_fname=audio_file_path)

#endregion


#region Part 5 Test to Speech - UI => streamlit

import edge_tts as tts
import streamlit as st
from edge_tts import communicate
from edge_tts.communicate import Communicate

st.sidebar.title("تنظیمات صدا")

voice = st.sidebar.selectbox("صدا", options=["fa-IR-DilaraNeural", "fa-IR-FaridNeural"])
rate = st.sidebar.slider("سرعت", min_value=-50, max_value=50, value=0, format="%d%%")
pitch = st.sidebar.slider("بم بودن صدا", min_value=-50, max_value=50, value=10, format="%dHz")
volume = st.sidebar.slider("بندی صدا", min_value=-50, max_value=50, value=0, format="%d%%")

file = st.file_uploader("فایل متنی", type="txt")

if file and st.button("تیدیل به ویس"):
    text = file.read().decode("utf-8")

    progress = st.progress(0, text="0%")
    progress.progress(30, text="30%")

    communicate = Communicate(
        text=text,
        voice=voice,
        rate=f"{rate:+d}%",
        volume=f"{volume:+d}%",
        pitch=f"{pitch:+d}Hz",
    )
    progress.progress(70, text="70%")
    communicate.save_sync(
        "output.mp3"
    )
    progress.progress(100, text="100%")

    st.audio("output.mp3")

#endregion















