import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

st.set_page_config(page_title="Churn Analytics", page_icon="🏦", layout="wide")

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as f: le_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f: ohe_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler

model, le_gender, ohe_geo, scaler = load_resources()

# Session state defaults
for k, v in [('geography',None),('gender',None),('age',35),('credit_score',650),
             ('balance',50000.),('estimated_salary',50000.),('tenure',5),
             ('num_of_products',1),('has_cr_card',0),('is_active_member',0),
             ('prediction_prob',None),('clear_form',False)]:
    if k not in st.session_state: st.session_state[k] = v

# Clear form if flag is set (before widgets render)
if st.session_state.clear_form:
    for k in ['geography','gender','age','credit_score','balance','estimated_salary',
              'tenure','num_of_products','has_cr_card','is_active_member']:
        st.session_state[k] = None if k in ['geography','gender'] else (35 if k=='age' else 650 if k=='credit_score' else 50000. if k in ['balance','estimated_salary'] else 5 if k=='tenure' else 1 if k=='num_of_products' else 0)
    st.session_state.clear_form = False

# Validation
def get_errors():
    err=[]
    req = {'Country':st.session_state.geography,'Gender':st.session_state.gender,'Age':st.session_state.age,
           'Credit Score':st.session_state.credit_score,'Account Balance':st.session_state.balance,
           'Estimated Salary':st.session_state.estimated_salary,'Tenure':st.session_state.tenure,
           'Number of Products':st.session_state.num_of_products,'Has Credit Card':st.session_state.has_cr_card,
           'Is Active Member':st.session_state.is_active_member}
    for name,val in req.items():
        if val is None or (isinstance(val,(int,float)) and np.isnan(val)): err.append(f"Please enter {name}.")
    for name,(val,lo,hi) in [('Age',(st.session_state.age,18,92)),('Credit Score',(st.session_state.credit_score,300,850)),
                             ('Account Balance',(st.session_state.balance,0,300000)),('Estimated Salary',(st.session_state.estimated_salary,0,200000)),
                             ('Tenure',(st.session_state.tenure,0,10)),('Number of Products',(st.session_state.num_of_products,1,4))]:
        if val is not None and not (lo <= val <= hi): err.append(f"{name} must be between {lo} and {hi}.")
    return err

st.title('🏦 Customer Churn Intelligence Portal')

with st.form("churn_form"):
    col1,col2,col3 = st.columns(3)
    with col1:
        st.subheader("📍 Geography")
        st.selectbox('Select Country', options=list(ohe_geo.categories_[0]), key='geography')
        st.selectbox('Gender', options=list(le_gender.classes_), key='gender')
        st.slider('Age', 18, 92, key='age')
    with col2:
        st.subheader("💰 Account Details")
        st.number_input('Credit Score', 300, 850, key='credit_score')
        st.number_input('Account Balance', 0.0, 300000.0, key='balance')
        st.number_input('Estimated Salary', 0.0, 200000.0, key='estimated_salary')
    with col3:
        st.subheader("⚙️ Behavioral Data")
        st.slider('Tenure (Years)', 0, 10, key='tenure')
        st.slider('Number of Products', 1, 4, key='num_of_products')
        st.selectbox('Has Credit Card?', [0,1], format_func=lambda x: 'Yes' if x else 'No', key='has_cr_card')
        st.selectbox('Is Active Member?', [0,1], format_func=lambda x: 'Yes' if x else 'No', key='is_active_member')
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    errors = get_errors()
    if errors:
        for e in errors: st.error(f"⚠️ {e}")
    else:
        try:
            input_data = pd.DataFrame({
                'CreditScore':[st.session_state.credit_score],
                'Gender':[le_gender.transform([st.session_state.gender])[0]],
                'Age':[st.session_state.age],
                'Tenure':[st.session_state.tenure],
                'Balance':[st.session_state.balance],
                'NumOfProducts':[st.session_state.num_of_products],
                'HasCrCard':[st.session_state.has_cr_card],
                'IsActiveMember':[st.session_state.is_active_member],
                'EstimatedSalary':[st.session_state.estimated_salary]
            })
            geo = ohe_geo.transform([[st.session_state.geography]]).toarray()
            geo_df = pd.DataFrame(geo, columns=ohe_geo.get_feature_names_out(['Geography']))
            input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)
            with st.spinner("Generating prediction..."):
                prob = model.predict(scaler.transform(input_data), verbose=0)[0][0]
            st.session_state.prediction_prob = float(prob)
            st.session_state.clear_form = True
            st.rerun()
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}. Please try again or contact support.")

if st.session_state.prediction_prob is not None:
    prob = st.session_state.prediction_prob
    st.divider()
    c1,c2 = st.columns(2)
    c1.metric("Churn Probability", f"{prob:.2%}")
    (c2.error if prob > 0.5 else c2.success)("High Risk: Customer is likely to churn." if prob > 0.5 else "Low Risk: Customer is likely to stay.")