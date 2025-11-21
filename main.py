# main.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page configuration
st.set_page_config(
    page_title="IMDB Movie Review Classifier",
    page_icon="🎬",
    layout="wide"
)


st.title("IMDB Movie Review Classifier by Manusha")

# Load model and word index
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model_imdb.h5')
        return model
    except:
        st.error("Model file 'model_imdb.h5' not found. Please train the model first.")
        return None

@st.cache_resource
def load_word_index():
    word_index = imdb.get_word_index()
    reverse_word_index = {value+3: key for (key, value) in word_index.items()}
    reverse_word_index[0] = '<PAD>'
    reverse_word_index[1] = '<START>'
    reverse_word_index[2] = '<UNK>'
    reverse_word_index[3] = 'the'
    return reverse_word_index

# Load test data
@st.cache_data
def load_test_data():
    (_, _), (x_test, y_test) = imdb.load_data(num_words=10000)
    x_test = pad_sequences(x_test, maxlen=256)
    return x_test, y_test

# Function to decode review
def decode_review(seq, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in seq if i != 0])

# Main application
def main():
    # Load resources
    model = load_model()
    reverse_word_index = load_word_index()
    x_test, y_test = load_test_data()
    
    if model is None:
        return
    
    st.header("5 Movie Reviews and Classification Results")
    st.write("Showing predictions for 5 sample reviews from the IMDB test dataset:")
    
    # Display 5 movie reviews with predictions
    for i in range(5):
        seq = x_test[i]
        text = decode_review(seq, reverse_word_index)
        actual_label = y_test[i]
        
        # Make prediction
        prob = model.predict(np.array([seq]), verbose=0)[0,0]
        predicted_label = 1 if prob >= 0.5 else 0
        
        # Display in columns
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"Review {i+1}")
            st.write(f"**Review Text:**")
            st.write(f"{text[:400]}..." if len(text) > 400 else text)
        
        with col2:
            actual_text = "Positive" if actual_label == 1 else "Negative"
            predicted_text = "Positive" if predicted_label == 1 else "Negative"
            
            st.write("**Results:**")
            st.write(f"Actual: {actual_text}")
            st.write(f"Predicted: {predicted_text}")
            st.write(f"Confidence: {prob:.2%}" if predicted_label == 1 else f"Confidence: {1-prob:.2%}")
            
            # Show correct/incorrect
            if actual_label == predicted_label:
                st.success("✓ Correct Prediction")
            else:
                st.error("✗ Incorrect Prediction")
        
        st.markdown("---")

# Run the app
if __name__ == "__main__":
    main()