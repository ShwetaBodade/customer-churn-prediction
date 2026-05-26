import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# === Page Configuration ===
# Set Streamlit page configuration for a wider layout and a more descriptive title
st.set_page_config(
    layout="wide",
    page_title="Customer Churn Dashboard",
    page_icon="📊"
)

# === Custom CSS Injection ===
# We'll add some custom CSS to make the UI elements more beautiful
# This now includes support for Streamlit's dark theme
st.markdown("""
<style>
    /* --- Light Theme --- */
    
    /* Main app background */
    .main {
        background-color: #F5F5F5;
    }
    
    /* KPI Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease-in-out;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    /* Style for the metric label */
    div[data-testid="stMetric"] label {
        font-size: 1.1rem;
        font-weight: 500;
        color: #555555;
    }
    
    /* Style for the metric value */
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5; /* A nice blue color */
    }

    /* Style for containers, forms, and expanders */
    div[data-testid="stVerticalBlock"] div[data-testid="stForm"],
    div[data-testid="stVerticalBlock"] div[data-testid="stExpander"],
    div.st-emotion-cache-1r4qj8v { /* This targets st.container(border=True) */
        background-color: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* Clean up expander header */
    div[data-testid="stExpander"] summary {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
    }
    
    /* Style tabs */
    div[data-testid="stTabs"] {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    button[data-testid="stTab"] {
        font-size: 1rem;
        font-weight: 600;
        color: #555;
    }
    
    button[data-testid="stTab"][aria-selected="true"] {
        color: #42A5F5;
        border-bottom: 3px solid #42A5F5;
    }
    
</style>
""", unsafe_allow_html=True)


# --- 1. Data Loading and Cleaning (Caching) ---
@st.cache_data
def load_data(filepath):
    """Loads and preprocesses the Telco Churn dataset."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found.")
        st.info("Please make sure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the same directory as the script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

    # Data Cleaning
    # Convert TotalCharges to numeric, coercing errors (blank spaces) to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Map SeniorCitizen 0/1 to No/Yes for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Rename columns to match expected names (if needed)
    df.rename(columns={
        'gender': 'Gender',
        'tenure': 'TenureMonths',
        'customerID': 'CustomerID'
    }, inplace=True)
    
    return df

# --- 2. Model Training (Caching) ---
@st.cache_resource
def train_model(df):
    """Preprocesses data and trains a RandomForest model."""
    
    # Handle missing TotalCharges (which we converted to NaN)
    if 'TotalCharges' in df.columns:
        median_total_charges = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)

    target = 'Churn'
    # Drop CustomerID as it's just an identifier
    X = df.drop(columns=[target, 'CustomerID'])
    y = df[target].map({'Yes': 1, 'No': 0})
    
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_pipeline.fit(X_train, y_train)
    
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
    }
    metrics['fpr'], metrics['tpr'], _ = roc_curve(y_test, y_pred_proba)
    metrics['roc_auc'] = auc(metrics['fpr'], metrics['tpr'])
    
    try:
        # Get feature names after one-hot encoding
        ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(ohe_feature_names)
        importances = model_pipeline.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame(sorted(zip(importances, all_feature_names), reverse=True), columns=['Importance', 'Feature'])
    except Exception as e:
        st.warning(f"Could not retrieve feature importance: {e}")
        feature_importance_df = pd.DataFrame(columns=['Importance', 'Feature'])

    # Return all columns used for prediction, and the original X for input options
    return model_pipeline, metrics, X_test, y_test, feature_importance_df, X.columns, X

# Load data
df = load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Stop execution if data loading failed
if df is None:
    st.stop()

# Train model
model, metrics, X_test, y_test, feature_importance_df, original_cols, X_for_inputs = train_model(df.copy())

# --- 3. Streamlit UI ---

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📊 Exploratory Data Analysis", "📈 Model Performance", "🔮 Make a Prediction"]
)
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("This dashboard provides an analysis of customer churn and a predictive model to identify at-risk customers. Built with Streamlit.")

# --- Page: Overview ---
if page == "🏠 Overview":
    st.title("Customer Churn Dashboard")
    st.markdown("### Welcome to the central hub for churn analysis")
    st.markdown("""
    This interactive dashboard is designed to help understand the factors that lead to customer churn
    and to predict which customers are most likely to leave.
    Use the navigation panel on the left to explore different sections.
    """)
    
    st.markdown("---")
    st.header("Key Performance Indicators (KPIs)")
    
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Customers", f"{len(df)}")
    kpi_cols[1].metric("Overall Churn Rate", f"{df['Churn'].value_counts(normalize=True)['Yes']:.1%}")
    kpi_cols[2].metric("Model Accuracy", f"{metrics['accuracy']:.1%}")
    kpi_cols[3].metric("Model F1-Score", f"{metrics['f1']:.1%}")
    
    st.markdown("---")
    
    # Use an expander for the raw data
    with st.expander("Raw Data Preview", expanded=False):
        st.dataframe(df.head(10))

# --- Page: Exploratory Data Analysis (EDA) ---
elif page == "📊 Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("Visualizing the distribution of customer data and its relationship with churn.")

    # Use tabs for a clean layout
    tab1, tab2, tab3 = st.tabs(["Demographics & Churn", "Contracts & Services", "Financials & Tenure"])

    with tab1:
        st.header("Churn and Demographic Distributions")
        col1, col2 = st.columns(2)
        with col1:
            fig_churn = px.pie(df, names='Churn', title='Overall Churn Distribution', hole=0.3, 
                               color_discrete_map={'Yes':'#EF5350', 'No':'#42A5F5'})
            fig_churn.update_layout(legend_title_text='Churn')
            st.plotly_chart(fig_churn, width='stretch')
        with col2:
            fig_gender = px.pie(df, names='Gender', title='Gender Distribution', hole=0.3)
            fig_gender.update_layout(legend_title_text='Gender')
            st.plotly_chart(fig_gender, width='stretch')
        
        col3, col4 = st.columns(2)
        with col3:
            fig_partner = px.histogram(df, x='Partner', color='Churn', barmode='group', 
                                        title='Churn by Partner Status',
                                        color_discrete_map={'Yes':'#EF5350', 'No':'#42A5F5'})
            st.plotly_chart(fig_partner, width='stretch')
        with col4:
            fig_dependents = px.histogram(df, x='Dependents', color='Churn', barmode='group', 
                                         title='Churn by Dependents Status',
                                         color_discrete_map={'Yes':'#EF5350', 'No':'#42A5F5'})
            st.plotly_chart(fig_dependents, width='stretch')
            
    with tab2:
        st.header("Service Agreements and Subscriptions")
        col1, col2 = st.columns(2)
        with col1:
            fig_contract = px.histogram(df, x='Contract', color='Churn', barmode='group', 
                                        title='Churn by Contract Type',
                                        color_discrete_map={'Yes':'#EF5350', 'No':'#42A5F5'})
            st.plotly_chart(fig_contract, width='stretch')
        with col2:
            fig_internet = px.histogram(df, x='InternetService', color='Churn', barmode='group', 
                                         title='Churn by Internet Service',
                                         color_discrete_map={'Yes':'#EF5350', 'No':'#42A5F5'})
            st.plotly_chart(fig_internet, width='stretch')

    with tab3:
        st.header("Financials and Customer Tenure")
        col1, col2 = st.columns(2)
        with col1:
            fig_scatter = px.scatter(df, x='TenureMonths', y='MonthlyCharges', color='Churn',
                                     title='Tenure vs. Monthly Charges by Churn',
                                     color_discrete_map={'Yes':'#EF5350', 'No':'#42A5F5'})
            st.plotly_chart(fig_scatter, width='stretch')
        with col2:
            fig_payment = px.histogram(df, x='PaymentMethod', color='Churn', barmode='group', 
                                   title='Churn by Payment Method',
                                   color_discrete_map={'Yes':'#EF5350', 'No':'#42A5F5'})
            fig_payment.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_payment, width='stretch')

# --- Page: Model Performance ---
elif page == "📈 Model Performance":
    st.title("Model Performance Evaluation")
    st.markdown("Assessing the accuracy and reliability of the churn prediction model.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Score": [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
            })
            st.dataframe(metrics_df.style.format({"Score": "{:.2%}"}), width='stretch')
        
        with st.container(border=True, height=500):
            st.subheader("Confusion Matrix")
            fig_cm = px.imshow(metrics['cm'], text_auto=True, 
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
                               color_continuous_scale='Blues',
                               title="Confusion Matrix")
            fig_cm.update_layout(height=400) # Ensure it fits
            st.plotly_chart(fig_cm, width='stretch')
    
    with col2:
        with st.container(border=True, height=830): # Match height
            st.subheader("Feature Importance")
            st.markdown("Top factors influencing the model's predictions.")
            fig_fi = px.bar(feature_importance_df.head(15), x='Importance', y='Feature', 
                            orientation='h', title='Top 15 Most Important Features')
            fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, height=750)
            st.plotly_chart(fig_fi, width='stretch')

    with st.container(border=True):
        st.subheader("ROC Curve")
        st.markdown("Receiver Operating Characteristic (ROC) Curve shows the trade-off between true positive rate and false positive rate.")
        fig_roc = px.area(
            x=metrics['fpr'], y=metrics['tpr'],
            title=f'ROC Curve (AUC = {metrics["roc_auc"]:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
        )
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_layout(height=500)
        st.plotly_chart(fig_roc, width='stretch')

# --- Page: Make a Prediction ---
elif page == "🔮 Make a Prediction":
    st.title("Live Churn Prediction")
    st.markdown("Enter a customer's details below to predict their likelihood of churning.")

    # Use a form for inputs
    with st.form("prediction_form"):
        st.header("Enter Customer Details")
        
        # Section 1: Customer Info
        st.subheader("👤 Customer Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gender = st.selectbox("Gender", options=X_for_inputs['Gender'].unique(), help="Select the customer's gender.")
        with col2:
            senior_citizen = st.selectbox("Senior Citizen", options=X_for_inputs['SeniorCitizen'].unique(), help="Is the customer a senior citizen?")
        with col3:
            partner = st.selectbox("Has Partner", options=X_for_inputs['Partner'].unique(), help="Does the customer have a partner?")
        with col4:
            dependents = st.selectbox("Has Dependents", options=X_for_inputs['Dependents'].unique(), help="Does the customer have dependents?")

        # Section 2: Service Subscriptions
        st.subheader("📡 Service Subscriptions")
        col1, col2, col3 = st.columns(3)
        with col1:
            phone_service = st.selectbox("Phone Service", options=X_for_inputs['PhoneService'].unique())
            multiple_lines = st.selectbox("Multiple Lines", options=X_for_inputs['MultipleLines'].unique())
        with col2:
            internet_service = st.selectbox("Internet Service", options=X_for_inputs['InternetService'].unique())
            online_security = st.selectbox("Online Security", options=X_for_inputs['OnlineSecurity'].unique())
            online_backup = st.selectbox("Online Backup", options=X_for_inputs['OnlineBackup'].unique())
        with col3:
            device_protection = st.selectbox("Device Protection", options=X_for_inputs['DeviceProtection'].unique())
            tech_support = st.selectbox("Tech Support", options=X_for_inputs['TechSupport'].unique())
            streaming_tv = st.selectbox("Streaming TV", options=X_for_inputs['StreamingTV'].unique())
            streaming_movies = st.selectbox("Streaming Movies", options=X_for_inputs['StreamingMovies'].unique())

        # Section 3: Billing & Contract
        st.subheader("💲 Billing & Contract")
        col1, col2, col3 = st.columns(3)
        with col1:
            contract = st.selectbox("Contract", options=X_for_inputs['Contract'].unique(), help="What is the customer's contract type?")
        with col2:
            paperless_billing = st.selectbox("Paperless Billing", options=X_for_inputs['PaperlessBilling'].unique())
        with col3:
            payment_method = st.selectbox("Payment Method", options=X_for_inputs['PaymentMethod'].unique())

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, help="How many months has the customer been with the company?")
        with col2:
            monthly_charges = st.slider("Monthly Charges ($)", min_value=float(X_for_inputs['MonthlyCharges'].min()), max_value=float(X_for_inputs['MonthlyCharges'].max()), value=float(X_for_inputs['MonthlyCharges'].mean()), step=0.01)
        with col3:
            max_total = float(X_for_inputs['TotalCharges'].max())
            total_charges = st.slider("Total Charges ($)", min_value=0.0, max_value=max_total, value=float(X_for_inputs['TotalCharges'].mean()), step=0.01)


        st.markdown("---")
        # Submit button
        submitted = st.form_submit_button("Predict Churn", type="primary", width='stretch')

    if submitted:
        # Create a DataFrame from the inputs
        input_data_dict = {
            'Gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'TenureMonths': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }
        
        # Ensure all columns expected by the model are present
        input_data = pd.DataFrame(input_data_dict)
        
        # Reorder columns to match model's training data
        input_data = input_data[original_cols]
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            churn_probability = prediction_proba[1] # Probability of Churn (Class 1)

            st.markdown("---")
            st.header("Prediction Result")
            
            with st.container(border=True):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if prediction == 1:
                        st.error("### LIKELY TO CHURN")
                        st.metric("Churn Probability", f"{churn_probability:.1%}")
                    else:
                        st.success("### LIKELY TO STAY")
                        st.metric("Retention Probability", f"{1-churn_probability:.1%}")
                
                with col2:
                    st.markdown(f"**Churn Risk Score: {churn_probability:.2f}**")
                    st.progress(churn_probability)
                    st.markdown(f"""
                    A score closer to **1.0** indicates a high risk of churn, while a score closer to **0.0** indicates a high likelihood of retention.
                    """)
            
            with st.expander("Show Input Data"):
                st.dataframe(input_data)

            # --- NEW: Add contextual graphs ---
            st.markdown("---")
            st.header("Prediction Context")
            st.markdown("How your inputs compare to historical churn data.")

            col1, col2 = st.columns(2)

            with col1:
                # Chart 1: Churn Rate by Contract Type
                st.subheader("Churn Rate by Contract Type")
                
                # Calculate churn rate by contract
                df_contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).loc[:, 'Yes'].reset_index(name='ChurnRate')
                
                # Add a color column to highlight the user's selection
                df_contract_churn['Color'] = df_contract_churn['Contract'].apply(lambda x: 'Your Selection' if x == contract else 'Other Contracts')
                
                fig_contract_comp = px.bar(
                    df_contract_churn, 
                    x='Contract', 
                    y='ChurnRate',
                    title='Churn Rate by Contract Type',
                    color='Color',
                    color_discrete_map={
                        'Your Selection': '#EF5350', # Red for selection
                        'Other Contracts': '#42A5F5'  # Blue for others
                    },
                    labels={'ChurnRate': 'Churn Rate', 'Contract': 'Contract Type'}
                )
                fig_contract_comp.update_yaxes(tickformat=".0%")
                fig_contract_comp.update_layout(showlegend=False) # Hide legend for simplicity
                st.plotly_chart(fig_contract_comp, width='stretch')
                st.markdown(f"Your selected contract type, **{contract}**, is highlighted. Month-to-month contracts have a significantly higher churn rate.")

            with col2:
                # Chart 2: Churn Distribution by Monthly Charges
                st.subheader("Churn by Monthly Charges")
                
                fig_charges_comp = px.histogram(
                    df, 
                    x='MonthlyCharges', 
                    color='Churn', 
                    nbins=50, 
                    title='Churn Distribution by Monthly Charges', 
                    barmode='overlay',
                    histnorm='percent',
                    color_discrete_map={'Yes':'#EF5350', 'No':'#42A5F5'}
                )
                
                # Add a vertical line for the user's input
                fig_charges_comp.add_vline(
                    x=monthly_charges, 
                    line_width=3, 
                    line_dash="dash", 
                    line_color="white", # Use white for dark mode visibility
                    annotation_text="Your Charge"
                )
                fig_charges_comp.update_layout(legend_title_text='Churn')
                st.plotly_chart(fig_charges_comp, width='stretch')
                st.markdown(f"Your input of **${monthly_charges:.2f}** (dashed line) is shown against the historical data.")
            # --- End of new section ---

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")