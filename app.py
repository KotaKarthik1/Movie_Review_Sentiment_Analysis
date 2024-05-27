from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your trained model (replace 'LRmodel.pkl' with your actual model file)
model = joblib.load('LRmodel.pkl')  # Load using joblib for efficiency
tfidf_vectorizer = joblib.load('tfid_vectorizer.pkl')

def preprocess_review(review):
    """Preprocesses the review text for prediction."""
    # review_vectorized = tfidf_vectorizer.transform([review])  # Use the extracted vectorizer
    # return review_vectorized
    # ... (in your training script)
    review_scale=tfidf_vectorizer.transform([review]).toarray()
    return review_scale

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        
        review_vectorized = preprocess_review(review)

        # Predict sentiment
        prediction = model.predict(review_vectorized)
        
        # Use np.squeeze to remove single-dimensional entries from the shape of an array.
        # prediction = np.squeeze(prediction)
        if prediction[0]==1:
            prediction="Negative"
        else:
            prediction="Positive"
        return render_template('result.html', sentiment=prediction, review=review)

    # For GET requests, just render the initial form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
