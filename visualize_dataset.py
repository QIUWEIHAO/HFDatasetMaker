import streamlit as st
from datasets import load_from_disk, Image
import pandas as pd
import base64
from io import BytesIO
from PIL import Image as PILImage

# ✅ 加载数据集
dataset = load_from_disk("processed_dataset")
df = dataset.to_pandas()

# ✅ 直接解码 `datasets.Image`
def decode_image(image_data):
    if image_data is None:
        return None
    try:
        return Image().decode_example(image_data)
    except Exception as e:
        st.warning(f"⚠️ 无法解析 `datasets.Image()` 数据: {e}")
        return None  # 遇到错误时返回 `None`

df["image"] = df["image"].apply(decode_image)  # ✅ 直接替代原 `image` 列

# ✅ 让 `image` 在 `st.markdown()` 里显示 HTML `<img>`
def image_to_html(img, index):
    if img is None:
        return "No Image"
    try:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f'<a href="data:image/png;base64,{img_base64}" target="_blank">' \
               f'<img src="data:image/png;base64,{img_base64}" width="400"></a>'
    except Exception as e:
        st.warning(f"⚠️ 无法处理图片: {e}")
        return "Error Image"

df["image_html"] = [image_to_html(img, i) for i, img in enumerate(df["image"])]  # ✅ 创建新的 HTML 列

# ✅ 显示数据表格（避免 `st.dataframe()` 直接渲染 `PIL.Image`）
st.write("### Hugging Face Dataset 预览")
st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
