import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import base64
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def count_distinct_customers(df):
    return len(df['CustomerID'].unique())

def proportion_high_churn_likelihood(df):
    high_churn_df = df[df['Churn_Prediction_Probability'] >= 0.8]
    proportion = len(high_churn_df) / len(df) * 100
    return proportion

def correlation_analysis(df):
    numeric_columns = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_columns.corr()

    st.subheader("Correlation Analysis")
    st.write("A correlation matrix helps identify which input features are most closely correlated with the target prediction variable i.e. which columns in the dataset are correlated with churn. Here's the correlation matrix:")

    # Create a heatmap to visualize the correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Explanation")

    st.write("Correlation values range from -1 to 1, with -1 denoting a perfect negative linear relationship, 0 indicating no linear relationship, and 1 signifying a perfect positive linear relationship. The closer the value is to -1 or 1, the stronger the linear relationship. In the matrix, we observe the most correlated feature with churn is:")

    # Find the 2 most highly correlated features with churn
    churn_correlation = correlation_matrix['Churn'].drop('Churn')
    top_correlated_features = churn_correlation.abs().sort_values(ascending=False).head(1)

    for feature, correlation in top_correlated_features.items():
        st.write(f"{feature}: {correlation:.2f}")


def main():
    
    st.set_page_config(layout="wide")
    
    st.title("User Churn Prediction Model")
    st.subheader("Predict customer churn using Machine Learning (Gradient Boosting via [XGBoost](https://xgboost.readthedocs.io/en/latest/get_started.html))")
    st.write("Churn prediction is vital for any mobile app business line that relies on repeat transaction volume. Leveraging machine learning in the task of churn prediction enables timely interventions to reduce churn rates. [Bain & Company shows](https://www.bain.com/insights/retaining-customers-is-the-real-challenge/) that a 5% rise in retention can boost profits by 25-95%. According to the [Harvard Business Review](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers), retaining existing customers can be 5-25 times cheaper than acquiring new ones.")
    st.write("Gradient Boosted Trees are an ensemble machine learning technique that combines multiple decision trees to create a predictive model. It does this by sequentially adding trees to the ensemble, with each new tree trained to correct the errors of the previous ones, improving overall model accuracy. XGBoost, or Extreme Gradient Boosting, is an implementation of gradient boosting. Below is a demonstration of XGBoost in the task of predicting which customers from a  sample [Kaggle dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) are most likely to churn.") 
    st.write("To run the demo download the input Kaggle datafile [here](https://docs.google.com/spreadsheets/d/1eDbLsDLFHl5VwJ0R95HqemwW3Os1LwhU/edit?usp=sharing&ouid=109058335558238702107&rtpof=true&sd=true) and upload it to generate predictions below. While the Kaggle sample data is e-commerce oriented, a very similar dataset and predictive model could be achieved by implementing [Mixpanel](https://docs.mixpanel.com/docs/getting-started/what-is-mixpanel) to aggregate in-app clickstream data. Such clickstream data could in turn could be used to train an XGBoost GBT to predict which customers are most likely to uninstall the app or to significantly reduce their app usage. Learn more [about me](https://drive.google.com/file/d/1FxZlr1P8I1FcJ4gXB0QX42bK9x-dJdB7/view?usp=sharing). Send me an email ebarn001@ucr.edu or call me at 1-619-862-4634 if you have any questions. Created by Eric Barnes, Machine Learning Research Engineer and M.Sc. candidate in Applied Artificial Intelligence @ the University of San Diego.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Video Demo")

        # Open the local MP4 video file
        video_file = open("Ceasars_Churn_Prediction_Model_Demo.mp4", "rb")
        video_contents = video_file.read()

        # Display the video
        st.video(video_contents)

    with col2:
        st.subheader("Load Data & Get Predictions")
        # File Upload
        uploaded_file = st.file_uploader("Upload the Kaggle data file [here](https://docs.google.com/spreadsheets/d/1eDbLsDLFHl5VwJ0R95HqemwW3Os1LwhU/edit?usp=sharing&ouid=109058335558238702107&rtpof=true&sd=true)", type=["xlsx"])

        if uploaded_file is not None:
            # Load the trained XGBoost model
            model = joblib.load("xgboost_churn_model.joblib")

            # Read the uploaded Excel file
            try:
                df = pd.read_excel(uploaded_file, sheet_name='E Comm')
            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")
                return



            # Preprocess the uploaded data (similar to the original code)
            df.fillna(0, inplace=True)
            label_encoders = {}
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le

            correlation_analysis(df)
            
            X = df.drop(['CustomerID', 'Churn'], axis=1)

            # Predict churn probabilities using the trained model
            churn_probabilities = model.predict_proba(X)[:, 1]
            df['Churn_Prediction_Probability'] = churn_probabilities

            # Sort the DataFrame by churn probability in descending order
            sorted_data = df.sort_values(by='Churn_Prediction_Probability', ascending=False)
            
            # Create a DataFrame to display customer IDs and their associated predictions
            prediction_df = sorted_data[['CustomerID', 'Churn_Prediction_Probability']]

            # Count distinct customer IDs
            distinct_customers = count_distinct_customers(df)
            #st.write(f"Total Number of Distinct Customer IDs: {distinct_customers}")

            # Calculate the proportion of customers with high churn likelihood
            churn_proportion = proportion_high_churn_likelihood(df)
            
            # Create a pie chart
            pie_chart_data = pd.DataFrame({
                'Category': ['High Churn Likelihood', 'Low Churn Likelihood'],
                'Proportion': [churn_proportion, 100 - churn_proportion]
            })

            fig = px.pie(pie_chart_data, names='Category', values='Proportion', title='Churn Likelihood Proportion')
            st.plotly_chart(fig)
            
            # Display the DataFrame
            st.subheader("Customer Churn Predictions")
            st.write(f"Proportion of Customers with High Churn Likelihood >= 80%: {churn_proportion:.2f}%")
            
            
            st.write(prediction_df)
            

if __name__ == "__main__":
    main()
