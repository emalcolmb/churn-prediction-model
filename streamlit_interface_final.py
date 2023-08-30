import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import base64

# Load the pre-trained model
model = tf.keras.models.load_model('balanced_model.h5')

# Load the scaler and fit it on training data
scaler = StandardScaler()

# Load the training data
train_data = pd.read_csv('subscriber_data.csv')
train_features = train_data.drop(['subscriber_id', 'subscription_status', 'cancellationRequested'], axis=1)
scaler.fit(train_features)

def make_predictions(file):
    data = pd.read_csv(file)
    features = data.drop(['subscriber_id', 'subscription_status', 'cancellationRequested'], axis=1)
    scaled_features = scaler.transform(features)
    churn_percentage_predictions = model.predict(scaled_features) * 100
    
    result_df = pd.DataFrame({
        'subscriber_id': data['subscriber_id'],
        'churn-prediction_percentage': churn_percentage_predictions.flatten()
    })
    
    # Sort the DataFrame by 'churn-prediction_percentage' in descending order
    result_df = result_df.sort_values(by='churn-prediction_percentage', ascending=False)
    
    return result_df


# Function to generate a download link
def get_table_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'

# Function to generate a download link for files
def get_file_download_link(filename):
    with open(filename, 'rb') as file:
        contents = file.read()
    b64 = base64.b64encode(contents).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'

# Initialize predictions availability status
predictions_available = False
# ... (previous imports and code)
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import base64



# Streamlit UI
def main():
    global predictions_available
    st.set_page_config(layout="wide")
    
    st.title("Subscriber Churn Prediction Model")
    st.subheader("Predict Subscriber Churn using Machine Learning, a [Recurrent Neural Network](https://www.tensorflow.org/guide/keras/working_with_rnns) + Simulated Data")
    st.write("Learn more [about me](https://drive.google.com/file/d/1FxZlr1P8I1FcJ4gXB0QX42bK9x-dJdB7/view?usp=sharing). Send me an email ebarn001@ucr.edu or call me at 1-619-862-4634 if you have more questions. Created by Eric Barnes, Machine Learning Research Engineer and M.Sc. candidate in Applied Artificial Intelligence @ the University of San Diego.")
    st.write("Churn prediction is vital for any subscription businesses, particularly critical in supporting & maintaining cashflow. Leveraging machine learning in the task of churn prediction enables timely interventions to reduce churn rates, backed by [Bain & Company's insight](https://www.bain.com/insights/retaining-customers-is-the-real-challenge/) that a 5% rise in retention boosts profits by 25-95%. According to the [Harvard Business Review](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers), it costs 5-25 times more to acquire new customers than retain existing ones - showing that retaining existing customers is the number 1 priority of many subscription businesses. Above you can find a link to my resume. Below you can find a demo as well as links to a) download the simulated subscriber data I generated b)upload that to generate and download predictions c) if you wish you can download the model itself with its dependencies if you'd like to tinker with it.")

    col1, col2 = st.columns(2)

    with col1:
    
        st.subheader("Churn Prediction Model Demo")

        # Open the local MP4 video file
        video_file = open("Churn_Prediction_Model_Demo.mp4", "rb")
        video_contents = video_file.read()

        # Display the video
        st.video(video_contents)
        

    with col2:
    
        st.subheader("Download Artifacts")
    
        download_link = '[Get the simulated subscriber data CSV](https://drive.google.com/file/d/1TFZrz3uxjzsWMnkgIO_cLMgO-DONXzWq/view?usp=sharing)'
        st.markdown(download_link, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload the subscriber data for churn prediction", type="csv")
        

        if uploaded_file is not None:
            result_df = make_predictions(uploaded_file)
            predictions_available = True  # Update predictions availability status

            st.write("Churn Predictions:")
            st.write(result_df[['subscriber_id', 'churn-prediction_percentage']])
    
        st.subheader("Download Predictions")

        # Disable the download button until predictions are available
        if predictions_available:
            if st.button("Download Churn Predictions"):
                st.markdown(get_table_download_link(result_df, "churn_predictions"), unsafe_allow_html=True)
        else:
            st.warning("Upload the subscriber data you want to analyze before trying to download results")

        st.subheader("Download Model")
        if st.button("Download Model"):
            model_filename = "balanced_model.h5"
            model_link = get_file_download_link(model_filename)
            st.markdown(model_link, unsafe_allow_html=True)

            # Explanation about TensorFlow Sequential RNN and downloading model
            st.write("A TensorFlow Sequential RNN is a type of recurrent neural network architecture in TensorFlow that allows you to build sequences of layers where the output of one layer serves as the input to the next layer.")
            st.write("You can download the model file, including all relevant dependencies, by clicking the \"Download Model\" button. This provides a hyperlink to the \"balanced_model.h5\" file & its dependences, allowing you to save and utilize the trained model and its weights.")



        # Plotly chart with detailed explanation
        if predictions_available:
            st.subheader("Churn Prediction Distribution")
            st.write("This pie chart displays the distribution of churn prediction categories. Each category represents a range of churn prediction percentages. Churn prediction percentage indicates the likelihood of a subscriber churning. Higher values indicate a higher likelihood of churn.")
            
            # Define bins for categorizing churn prediction percentages
            bins = [0, 40, 100]  # Keep two relevant categories: '0-40%' and '40-100%'
            labels = ['Low Churn Risk', 'High Churn Risk']
            
            # Categorize churn prediction percentages into bins
            result_df['churn_category'] = pd.cut(result_df['churn-prediction_percentage'], bins=bins, labels=labels)
            
            # Calculate the distribution of categories
            category_counts = result_df['churn_category'].value_counts()
            
            # Create a pie chart using Plotly Express
            fig = px.pie(category_counts, 
                         names=category_counts.index, 
                         values=category_counts.values, 
                         title="Churn Prediction Distribution")
            
            # Customize the chart layout and style
            # Display category percentages as labels and adjust segment pulling
            fig.update_traces(textinfo='percent+label', pull=[0.2, 0], marker=dict(line=dict(color='#000000', width=2)))
            
            # Use a white background template for better readability
            fig.update_layout(template="plotly_white")
            
            # Show the interactive pie chart in the Streamlit app
            st.plotly_chart(fig)
            


if __name__ == '__main__':
    main()
