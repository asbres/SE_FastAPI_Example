import streamlit as st
from transformers import pipeline
from PIL import Image, ImageOps
import io

# Настройка страницы
st.set_page_config(page_title="Background Remover", layout="wide")

# Заголовок
st.title("Удаление фона с фото (RMBG-1.4)")
st.write("Модель: [BRIA RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
if uploaded_file:
    if st.button("Очистить кэш модели"):
        st.cache_resource.clear()

@st.cache_resource
def load_pipeline():
    return pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

pipe = load_pipeline()

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    with st.spinner("Обработка изображения..."):
        pillow_mask = pipe(image, return_mask=True)
        pillow_image = pipe(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Оригинал", use_container_width=True)

    with st.spinner("Убираем фон..."):
        result_image = pipe(image)

    with col2:
        st.image(pillow_image, caption="Изображение без фона", use_container_width=True)

    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    
    st.divider()

    st.download_button(
        label="📥 Скачать результат (PNG)",
        data=buf.getvalue(),
        file_name=f"no_bg_{uploaded_file.name.split('.')[0]}.png",
        mime="image/png",
        use_container_width=True
    )

else:
    st.info("Загрузите изображение, чтобы начать.")