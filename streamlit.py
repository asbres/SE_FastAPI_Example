import streamlit as st
from transformers import pipeline
from PIL import Image
import io

# Заголовок
st.title("Удаление фона с фото (RMBG-1.4)")
st.write("Модель: [BRIA RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)")


uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])


@st.cache_resource
def load_pipeline():
    return pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

pipe = load_pipeline()

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Оригинальное изображение", use_container_width=True)

    with st.spinner("Обработка изображения..."):
        pillow_mask = pipe(image, return_mask=True)
        pillow_image = pipe(image)

    st.subheader("Результаты сегментации")

    col1, col2 = st.columns(2)
    with col1:
        st.image(pillow_mask, caption="Маска сегментации", use_container_width=True)
    with col2:
        st.image(pillow_image, caption="Изображение без фона", use_container_width=True)

    buf = io.BytesIO()
    pillow_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="Скачать изображение без фона",
        data=byte_im,
        file_name="result.png",
        mime="image/png"
    )