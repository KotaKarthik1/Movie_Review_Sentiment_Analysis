from flask import Flask, render_template, request
import joblib  # For efficient model loading

app = Flask(__name__)

# Load your trained model 
model = joblib.load('LRmodel.pkl')  # Load the logistic regression model from the saved file
tfidf_vectorizer = joblib.load('tfid_vectorizer.pkl')  # Load the fitted TF-IDF vectorizer


def preprocess_review(review):
    """
    Preprocesses the review text for prediction.

    Args:
        review (str): The raw review text.

    Returns:
        numpy.ndarray: A vectorized representation of the review using TF-IDF.
    """
    review_scale = tfidf_vectorizer.transform([review]).toarray()  # Transform the review into a TF-IDF vector
    return review_scale


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route for handling GET and POST requests.

    GET: Renders the initial form (index.html).
    POST: Processes the review, makes a prediction, and renders the result (result.html).
    """
    if request.method == 'POST':
        review = request.form['review']  # Get the review text from the form
        review_vectorized = preprocess_review(review)  # Preprocess the review

        # Predict sentiment
        prediction = model.predict(review_vectorized)  # Predict the sentiment (0 or 1)
        
        # Convert numerical prediction to sentiment label
        if prediction[0] == 0:
            prediction = "Positive"
        else:
            prediction = "Negative"

        return render_template('result.html', sentiment=prediction, review=review)  # Render the result page

    # For GET requests, render the initial form
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode
