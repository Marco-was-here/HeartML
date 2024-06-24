import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
 
 
@st.cache
def load_data():
    data = pd.read_csv('heart.csv',header=0, sep=";")
    return data
 
 
def preprocess_input(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    sex = 0 if sex == 'female' else 1
    fbs = 1 if fbs == 'yes' else 0
    exang = 1 if exang == 'yes' else 0
    return {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
      }
 
 
data = load_data()
if 'target' not in data.columns:
    raise ValueError("The 'target' column is missing from the dataset.")
 
X = data.drop(columns=['target'])  # Drop 'target' column from features
y = data['target']  # Assign 'target' column to y
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
 
 
st.title('Heart Disease Prediction')
 
 
st.header('Enter Patient Details')
age = st.number_input('Age', 20, 100, 50)
sex = st.selectbox('Sex', ['male', 'female'])
cp = st.number_input('Chest Pain Type (cp)', 0, 3, 1)
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 90, 200, 120)
chol = st.number_input('Serum Cholesterol (mg/dl)', 100, 600, 200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['true', 'false'])
restecg = st.number_input('Resting Electrocardiographic Results (restecg)', 0, 2, 1)
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', 70, 220, 150)
exang = st.selectbox('Exercise Induced Angina (exang)', ['yes', 'no'])
oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', 0.0, 6.2, 1.0)
slope = st.number_input('Slope of the Peak Exercise ST Segment (slope)', 0, 2, 1)
ca = st.number_input('Number of Major Vessels Colored by Flourosopy (ca)', 0, 4, 1)
thal = st.number_input('Thalassemia (thal)', 0, 3, 1)
 
 
def predict(model, input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    return prediction, prediction_proba
 
 
if st.button('Predict'):
    input_data = preprocess_input(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    prediction, prediction_proba = predict(model, input_data)
    # Display prediction
    if prediction[0] == 0:
        st.error('The patient is likely to **not** have heart disease.')
    else:
        st.success('The patient is likely to have **heart disease**.')
 
    st.write(f'Prediction Probability: {prediction_proba[0][1]:.2f}')
