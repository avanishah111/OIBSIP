import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Read the dataset
df = pd.read_csv(r"C:\Users\avani\Downloads\archive (14)\spam_ham_dataset.csv")

# Data preprocessing and feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label_num']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Function to predict spam or ham
def predict_spam_or_ham(email_text):
    new_email_vectorized = vectorizer.transform([email_text])
    prediction = model.predict(new_email_vectorized)
    return prediction[0]

# Streamlit web app
def main():
    st.set_page_config(layout="wide")
    st.title("Spam Email Detection By Avani Shah")
    st.write("Enter your email below to check if it's spam or not (ham).")

    user_input = st.text_area("Enter an email:", "")
    if st.button("Check"):
        if user_input.strip():
            prediction = predict_spam_or_ham(user_input)
            if prediction == 1:
                st.write("The email is predicted as spam.")
            else:
                st.write("The email is predicted as not spam (ham).")
        else:
            st.write("Please enter an email before checking.")
    
    # Load and display the image on the right sidebar
    image = Image.open(r"E:\imp\avanipe.png")
    st.sidebar.image(image, caption="Avani Shah", use_column_width=True)
# Add your contact information at the bottom
    st.sidebar.markdown("---")
    st.sidebar.markdown("If you have any issues related to this site, feel free to contact me.")
    st.sidebar.markdown("Contact Avani Shah:")
    st.sidebar.markdown("Email: avani.work123@gmail.com")
    st.sidebar.markdown("LinkedIn: [Avani Shah](https://www.linkedin.com/in/avanishah111/)")

if __name__ == "__main__":
    main()
