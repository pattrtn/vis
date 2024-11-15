import streamlit as st
import joblib
import pandas as pd

# Load CRF model
# This function caches the loaded model to avoid reloading it multiple times during app execution.
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# Function to process input text with the CRF model
# Splits input text into tokens, generates features for each token, and predicts entities using the CRF model.
def predict_entities(crf_model, text):
    tokens = text.split()  # Tokenize the input text
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]  # Generate features for each token
    predictions = crf_model.predict([features])[0]  # Predict entities for the tokens
    return list(zip(tokens, predictions))  # Return a list of token-entity pairs

# Feature extraction function
# Generates features for each token, considering its properties and context (previous and next tokens).
def tokens_to_features(tokens, i):
    stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]  # Define common stopwords
    word = tokens[i]  # Current token

    # Basic features of the current token
    features = {
        "bias": 1.0,  # Bias term to improve model performance
        "word.word": word,  # Current word
        "word[:3]": word[:3],  # First three characters of the word
        "word.isspace()": word.isspace(),  # Checks if the word is whitespace
        "word.is_stopword()": word in stopwords,  # Checks if the word is a stopword
        "word.isdigit()": word.isdigit(),  # Checks if the word is numeric
        "word.islen5": word.isdigit() and len(word) == 5  # Checks if the word is a 5-digit number (postal code)
    }

    # Features for the previous token
    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,  # Previous word
            "-1.word.isspace()": prevword.isspace(),  # Checks if the previous word is whitespace
            "-1.word.is_stopword()": prevword in stopwords,  # Checks if the previous word is a stopword
            "-1.word.isdigit()": prevword.isdigit(),  # Checks if the previous word is numeric
        })
    else:
        features["BOS"] = True  # Marks the beginning of a sentence

    # Features for the next token
    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,  # Next word
            "+1.word.isspace()": nextword.isspace(),  # Checks if the next word is whitespace
            "+1.word.is_stopword()": nextword in stopwords,  # Checks if the next word is a stopword
            "+1.word.isdigit()": nextword.isdigit(),  # Checks if the next word is numeric
        })
    else:
        features["EOS"] = True  # Marks the end of a sentence

    return features

# Load data for dropdowns
# Reads data from the provided Excel file to populate dropdown options for districts, subdistricts, and provinces.
file_path = './thaidata.xlsx'
data = pd.read_excel(file_path, sheet_name='db')
tambon_options = [""] + data['TambonThaiShort'].dropna().unique().tolist()  # Subdistrict options with default
district_options = [""] + data['DistrictThaiShort'].dropna().unique().tolist()  # District options with default
province_options = [""] + data['ProvinceThai'].dropna().unique().tolist()  # Province options with default

# Map postal codes to district, subdistrict, and province
postal_code_mapping = data.set_index(['TambonThaiShort', 'DistrictThaiShort', 'ProvinceThai'])['PostCodeMain'].to_dict()

# Streamlit app setup
st.title("NER Model Visualization")
st.markdown(
    """This app allows you to visualize and interact with a Named Entity Recognition (NER) model
    trained for address extraction. Input the required fields and run the model to see predictions."""
)

# Load the CRF model from a predefined file path
model_file_path = './model.joblib'
model = load_model(model_file_path)  # Load the model
st.success("Model loaded successfully!")

# Input fields for address components
name = st.text_input("ชื่อ (Name):")  # Name field
address = st.text_input("ที่อยู่ (Address):")  # Address field
district = st.selectbox("ตำบล (District):", options=tambon_options)  # Dropdown for subdistricts
subdistrict = st.selectbox("อำเภอ (Sub-district):", options=district_options)  # Dropdown for districts
province = st.selectbox("จังหวัด (Province):", options=province_options)  # Dropdown for provinces

# Automatically determine postal code based on district, subdistrict, and province
postal_code = ""
if district and subdistrict and province:
    postal_code = postal_code_mapping.get((district, subdistrict, province), "")

st.text_input("รหัสไปรษณีย์ (Postal Code):", value=postal_code, disabled=True)  # Display postal code as a read-only field

# Run button
if st.button("Run"):
    # Combine all inputs into a single text for processing
    input_text = f"{name} {address} {district} {subdistrict} {province} {postal_code}"

    # Run predictions on the combined input text
    results = predict_entities(model, input_text)

    # Display prediction results in a table
    st.subheader("Prediction Results")
    result_df = pd.DataFrame(results, columns=["Token", "Entity"])

    # Add validation column with expected answers
    expected_answers = ["O", "O", "ADDR"] + ["ADDR"] * (len(result_df) - 5) + ["LOC", "LOC", "LOC", "POST"]
    # result_df["Validation"] = expected_answers[:len(result_df)]

    st.dataframe(result_df)

    # Visualization of predictions with color-coding
    st.subheader("Entity Visualization")
    for token, entity in results:
        color = (
            "#FFCCCB" if entity == "LOC" else  # Locations are highlighted in light red
            "#D3D3D3" if entity == "POST" else  # Postal codes are highlighted in grey
            "#ADD8E6" if entity == "ADDR" else  # Address elements are highlighted in light blue
            "#90EE90"  # All other tokens are highlighted in light green
        )
        st.markdown(f"<span style='background-color:{color}'>{token} ({entity})</span>", unsafe_allow_html=True)  # Inline styling for visualization
