import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (224,224)

class_names = ["Biodegradable","Hazardous","Recyclable"]

# disposal suggestions
suggestions = {
"Biodegradable":"Dispose in compost bin",
"Recyclable":"Dispose in recycling bin",
"Hazardous":"Dispose in hazardous waste bin"
}

# icons for each waste class
icons = {
"Biodegradable":"🌿",
"Recyclable":"♻️",
"Hazardous":"☣️"
}

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("waste_classifier.keras", compile=False, safe_mode=False)
    return model

model = load_model()

st.title("♻️ AI Waste Segregation Assistant")

uploaded_file = st.file_uploader("Upload Waste Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image",width="stretch")

    img = image.resize(IMG_SIZE)
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)

    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds)*100)

    # prediction with icon
    st.subheader(f"{icons[predicted_class]} Prediction: {predicted_class}")

    # confidence bar
    st.progress(int(confidence))

    st.write(f"Confidence: {confidence:.2f}%")

    # disposal suggestion
    st.success(f"Suggestion: {suggestions[predicted_class]}")

# footer
st.markdown("---")
st.caption("AI Waste Segregation Assistant | Deep Learning Project")