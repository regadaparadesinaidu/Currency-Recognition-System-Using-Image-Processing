import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# 🔊 NEW IMPORTS (voice)
from gtts import gTTS
import tempfile

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "currency_model.pth"
CLASS_FILE = "classes.txt"
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD CLASS NAMES (AUTO)
# -----------------------------
def load_classes():
    with open(CLASS_FILE, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

class_names = load_classes()

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)

    # IMPORTANT: same classifier as training
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model

model = load_model()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# -----------------------------
# 🔊 TEXT TO SPEECH FUNCTION
# -----------------------------
def speak(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# -----------------------------
# UI
# -----------------------------
st.title("💰 Currency Recognition App")

uploaded_file = st.file_uploader("Upload Currency Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, 1)

    predicted_class = class_names[pred.item()]
    confidence_score = confidence.item()

    st.success(f"💵 Prediction: {predicted_class}")
    st.info(f"🔥 Confidence: {confidence_score:.4f}")

    # 🔊 VOICE OUTPUT
    voice_text = f"The predicted currency is {predicted_class} with confidence {confidence_score:.2f}"
    audio_file = speak(voice_text)
    st.audio(audio_file)

    # Optional: Top-3 predictions 🔥
    st.subheader("Top 3 Predictions")
    top3_prob, top3_idx = torch.topk(probs, 3)

    for i in range(3):
        st.write(f"{class_names[top3_idx[0][i]]} → {top3_prob[0][i]:.4f}")