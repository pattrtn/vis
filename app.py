import streamlit as st
import joblib
import pandas as pd

# Load CRF model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# Function to process input text with the CRF model
def predict_entities(crf_model, text):
    tokens = text.split()
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    predictions = crf_model.predict([features])[0]
    return list(zip(tokens, predictions))

# Updated feature extraction with stopwords
def tokens_to_features(tokens, i):
    stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]
    word = tokens[i]

    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5
    }

    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True

    return features

# Streamlit app setup
st.title("NER Model Visualization")
st.markdown(
    """This app allows you to visualize and interact with a Named Entity Recognition (NER) model
    trained for address extraction. Input the required fields and run the model to see predictions."""
)

# Upload CRF model
model_file = st.file_uploader("Upload CRF model (.joblib file):", type="joblib")

if model_file:
    model = load_model(model_file)
    st.success("Model loaded successfully!")

    # Input fields
    name = st.text_input("ชื่อ (Name):")
    address = st.text_input("ที่อยู่ (Address):")
    district = st.text_input("ตำบล (District):")
    subdistrict = st.text_input("อำเภอ (Sub-district):")
    province = st.text_input("จังหวัด (Province):")
    postal_code = st.text_input("รหัสไปรษณีย์ (Postal Code):")

    # Run button
    if st.button("Run"):
        # Combine all inputs into a single text
        input_text = f"{name} {address} {district} {subdistrict} {province} {postal_code}"

        # Run predictions
        results = predict_entities(model, input_text)

        # Display results
        st.subheader("Prediction Results")
        result_df = pd.DataFrame(results, columns=["Token", "Entity"])
        st.dataframe(result_df)

        # Visualization
        st.subheader("Entity Visualization")
        for token, entity in results:
            color = "#FFCCCB" if entity == "LOC" else ("#D3D3D3" if entity == "POST" else "#90EE90")
            st.markdown(f"<span style='background-color:{color}'>{token} ({entity})</span>", unsafe_allow_html=True)
