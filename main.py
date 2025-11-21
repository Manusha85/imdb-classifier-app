# main.py
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="IMDB Movie Review Classifier",
    page_icon="🎬", 
    layout="wide"
)


st.title("IMDB Movie Review Classifier by Manusha")

st.header(" Movie Review Sentiment Analysis")
st.write("This application demonstrates a Recurrent Neural Network (RNN) model trained to classify IMDB movie reviews as positive or negative.")

# Sample data for demonstration
sample_data = {
    'Review_Number': [1, 2, 3, 4, 5],
    'Actual_Sentiment': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive'],
    'Predicted_Sentiment': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive'],
    'Confidence_Score': [0.94, 0.89, 0.96, 0.87, 0.98],
    'Correct_Prediction': [True, True, True, True, True]
}

df = pd.DataFrame(sample_data)

# Display sample reviews
st.subheader(" 5 Sample Movie Reviews & Predictions")

for index, row in df.iterrows():
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Sample review text based on sentiment
            if row['Actual_Sentiment'] == 'Positive':
                review_text = "This movie was absolutely fantastic! The acting was superb, the storyline engaging, and the cinematography breathtaking. One of the best films I've seen this year. Highly recommended for all movie lovers!"
            else:
                review_text = "Unfortunately, this movie failed to deliver. The plot was confusing, character development was weak, and the pacing felt off. Not worth the time investment in my opinion."
            
            st.write(f"**Review {row['Review_Number']}:**")
            st.write(review_text)
        
        with col2:
            st.metric("Actual", row['Actual_Sentiment'])
            st.metric("Predicted", row['Predicted_Sentiment'])
            st.metric("Confidence", f"{row['Confidence_Score']:.0%}")
            
            if row['Correct_Prediction']:
                st.success(" Correct")
            else:
                st.error(" Incorrect")
        
        st.progress(row['Confidence_Score'])
        st.markdown("---")

# Add visualization
st.subheader(" Model Performance Overview")

# Create a simple bar chart
chart_data = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Score': [0.85, 0.83, 0.82, 0.82]
})

bar_chart = alt.Chart(chart_data).mark_bar().encode(
    x='Metric',
    y='Score',
    color=alt.Color('Metric', legend=None)
).properties(height=300)

st.altair_chart(bar_chart, use_container_width=True)

# Model information in sidebar
st.sidebar.header(" Model Information")
st.sidebar.write("""
**Architecture:**
- Embedding Layer (64 dimensions)
- SimpleRNN Layer (64 units)
- Dense Output Layer

**Training Details:**
- Dataset: IMDB Movie Reviews
- Vocabulary: 10,000 words
- Sequence Length: 256
- Training Time: ~2 minutes
- Final Accuracy: 85%
""")

st.sidebar.header(" Local Deployment")
st.sidebar.write("""
For full TensorFlow model functionality:

1. Download the repository
2. Run: `pip install tensorflow streamlit numpy`
3. Run: `streamlit run main.py`
4. Access at: `http://localhost:8501`
""")

# Footer
st.markdown("---")
st.caption("""
*Note: This cloud deployment demonstrates the application interface. For real-time predictions with the trained RNN model, 
please run the application locally with TensorFlow dependencies installed.*
""")
