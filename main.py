import streamlit as st
from time import *

st.title("Загрузка фотографии")
uploaded_file = st.file_uploader(
    "**Загрузите фотографию**",
    type=["jpg", "jpeg", "png", "webp", ],
)
if uploaded_file is not None:
    st.image(uploaded_file, caption="Ваша фотография")
    st.success("Спасибо, красивая фотка ❤️")