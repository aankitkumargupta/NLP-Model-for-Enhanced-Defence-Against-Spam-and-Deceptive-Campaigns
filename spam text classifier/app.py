from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load models, vectorizer, and accuracies
models = {
    'Naive Bayes': joblib.load('best_naive_bayes.pkl'),
    'SVM': joblib.load('best_svm.pkl'),
    'Logistic Regression': joblib.load('best_logistic_regression.pkl'),
    'Random Forest': joblib.load('best_random_forest.pkl'),
    'KNN': joblib.load('best_knn.pkl')
}
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model_accuracies = joblib.load('model_accuracies.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            user_input = request.form['text']
            selected_model = request.form['model']
            model = models[selected_model]
            accuracy = model_accuracies.get(selected_model, "N/A")

            # Process and predict
            input_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(input_vectorized)
            result = 'Spam' if prediction[0] == 1 else 'Ham'
            result += f" (Accuracy: {accuracy*100:.2f}%)"  # Add accuracy to the result
        except Exception as e:
            result = f"Error: {e}"

        return render_template('index.html', result=result)
    
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
