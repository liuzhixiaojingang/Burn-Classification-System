import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# 设置页面配置
st.set_page_config(
    page_title="Burn Classification System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载预训练模型 - 使用相对路径
@st.cache_resource
def load_model():
    try:
        # 假设模型文件放在同一目录下的models文件夹
        model_path = os.path.join("rf.pkl")
        model = joblib.load(model_path)
        model.feature_names_in_ = ['BG1', 'Ascorbic acid', 'Pregnenolone sulfate', 'IL-1β', 
                                 '5-Methoxytryptamine', 'EGF', 'BG2']
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

# Burn type mapping (English labels)
burn_type_mapping = {
    0: "Normal",
    1: "Superficial partial-thickness",
    2: "Deep partial-thickness",
    3: "Full-thickness",
    4: "Electrical",
    5: "Flame"
}

# Page title
st.title("Burn Classification System")

# 加载图片 - 使用相对路径
try:
    # 假设图片放在同一目录下的images文件夹
    image_path = os.path.join("output.png")
    shap_image = Image.open(image_path)
except Exception as e:
    st.warning(f"Could not load feature importance image: {str(e)}")
    shap_image = None

# Input form
with st.form("input_form"):
    st.header("Input Burn Characteristics")
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.number_input("BG1", value=-1.89)
        feature2 = st.number_input("Ascorbic acid", value=14933.01)
        feature3 = st.number_input("Pregnenolone sulfate", value=4225.85)
        feature4 = st.number_input("IL-1β", value=388.72)
    with col2:
        feature5 = st.number_input("5-Methoxytryptamine", value=38206.77)
        feature6 = st.number_input("EGF", value=538.56)
        feature7 = st.number_input("BG2", value=-0.51)
    submitted = st.form_submit_button("Predict")

# Prediction and visualization
if submitted and model is not None:
    try:
        # Prepare input data
        input_data = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5, feature6, feature7]],
                                columns=model.feature_names_in_)
        
        # Get prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # === 创建两列布局 ===
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Probability Distribution
            st.subheader("Classification Probabilities")
            fig_prob, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(range(len(probabilities)), probabilities, 
                         color=['#1f77b4' if i != prediction else '#ff7f0e' for i in range(len(probabilities))])
            ax.set_xticks(range(len(probabilities)))
            ax.set_xticklabels([burn_type_mapping[i] for i in range(len(probabilities))], rotation=45)
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}',
                        ha='center', va='bottom')
            st.pyplot(fig_prob)
        
        with col2:
            # 2. 直接插入图片
            st.subheader("Multi-class Feature Importance")
            if shap_image is not None:
                st.image(shap_image, caption="Multi-class Feature Importance", use_column_width=True)
            else:
                st.warning("Feature importance image not available")
        
        # === 3. Input Feature Values ===
        st.subheader("Input Features")
        st.dataframe(input_data.style.format("{:.3f}").highlight_max(axis=1, color='#ff7f0e'))
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

elif submitted and model is None:
    st.error("Model not loaded - cannot make predictions")