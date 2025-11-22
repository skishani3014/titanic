import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Set page configuration for a professional look
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide", initial_sidebar_state="expanded")

# Load Model and Data 
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('titanic_model.pkl')
        df = pd.read_csv('titanic.csv')
        return model, df
    except FileNotFoundError:
        st.error("One or more required files (model.pkl or titanic.csv) are missing. Please ensure your project structure is correct and the model has been trained.")
        st.stop()

model, df = load_resources()

# Sidebar Navigation 
st.sidebar.title("🚢 Navigation")
page = st.sidebar.radio("Choose a section", ["Home", "Data Exploration", "Prediction", "Model Performance"])

# Content Sections
if page == "Home":
    st.markdown("<h1 style='text-align: center;'>🚢 Titanic Survival Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Welcome! This application predicts the survival of a passenger on the Titanic using a machine learning model. Use the sidebar to navigate through the app.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("titanic.jpg")

elif page == "Data Exploration":
    st.markdown("<h1 style='text-align: center;'>Data Exploration</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>A closer look at the raw Titanic dataset.</p>", unsafe_allow_html=True)

    # Dataset overview section
    st.markdown("<h3 style='text-align: center;'>Dataset at a Glance</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
    
    # Interactive visualizations
    st.markdown("<h3 style='text-align: center;'>Key Visualizations</h3>", unsafe_allow_html=True)
    
    # Plot 1: Survival Rate by Passenger Class
    survival_by_pclass = df.groupby('Pclass')['Survived'].value_counts(normalize=True).unstack().reset_index()
    survival_by_pclass.rename(columns={0: 'Did Not Survive', 1: 'Survived'}, inplace=True)
    fig1 = px.bar(
        survival_by_pclass,
        x='Pclass',
        y=['Did Not Survive', 'Survived'],
        title='Survival Rate by Passenger Class',
        labels={'value': 'Proportion', 'Pclass': 'Passenger Class'},
        barmode='group'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Plot 2: Age Distribution of Passengers
    fig2 = px.histogram(
        df,
        x='Age',
        color='Survived',
        marginal='box',
        title='Age Distribution by Survival Status',
        labels={'Age': 'Age of Passenger'},
        nbins=20
    )
    st.plotly_chart(fig2, use_container_width=True)
    
elif page == "Prediction":
    st.markdown("<h1 style='text-align: center;'>Predict Titanic Survival</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter the details of a passenger to see if our model predicts their survival.</p>", unsafe_allow_html=True)

    # Create a form for user inputs
    with st.form("prediction_form"):
        st.markdown("<h3 style='text-align: center;'>Passenger Information</h3>", unsafe_allow_html=True)
        
        # Two columns for a cleaner layout
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = Upper Class, 2 = Middle Class, 3 = Lower Class")
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.slider("Age", 0, 80, 30)
            sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
        
        with col2:
            parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
            fare = st.number_input("Fare ($)", 0.0, 500.0, 32.0, format="%.2f")
            embarked_port = st.selectbox("Port of Embarkation", ["S (Southampton)", "C (Cherbourg)", "Q (Queenstown)"])
            
        submit_button = st.form_submit_button(label="Predict Survival")

    if submit_button:
        # Map inputs to the model's expected format
        sex_encoded = 1 if sex == "male" else 0
        
        embarked_s = 1 if embarked_port.startswith("S") else 0
        embarked_q = 1 if embarked_port.startswith("Q") else 0
        
        # Create a DataFrame for the prediction
        input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_q, embarked_s]],
                                  columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S'])
        
        with st.spinner('Making prediction...'):
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
        
        st.write("---")
        st.markdown("<h3 style='text-align: center;'>Prediction Result</h3>", unsafe_allow_html=True)
        
        if prediction == 1:
            st.success(f"Prediction: This passenger would have **survived!** 🎉")
            st.write(f"Confidence: **{prediction_proba[1]*100:.2f}%**")
        else:
            st.error(f"Prediction: This passenger would **not have survived.** 😔")
            st.write(f"Confidence: **{prediction_proba[0]*100:.2f}%**")
            
elif page == "Model Performance":
    st.markdown("<h1 style='text-align: center;'>Model Performance</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Here are the performance metrics for the trained model on the test data.</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center;'>Model Evaluation Metrics</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>🎯 <b>Accuracy:</b> 85.00%</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>📈 <b>Cross-Validation Score:</b> 82.50%</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center;'>Confusion Matrix</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Note: This matrix is for the test data used during training.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("confusion_matrix.png")
