# Importing necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import string

# Downloading necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Loading the dataset
df = pd.read_csv('C:/Users/aanki/Desktop/Placement/spam_ham_dataset.csv')

# Preprocessing
def preprocess_text(text):
    # Removing punctuation and lowercase
    text = "".join([char.lower() for char in text if char not in string.punctuation])
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Encoding the target variable
encoder = LabelEncoder()
df['label_encoded'] = encoder.fit_transform(df['label'])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])
y = df['label_encoded']

# Spliting the data to train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123)

model_accuracies = {}

# Function to train and evaluate a model
def train_evaluate_model(model, params, X_train, y_train, X_test, y_test, model_name):
    print(f"Training and evaluating {model_name}...")
    try:
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name] = accuracy
        print(f"Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
        return best_model
    except Exception as e:
        print(f"An error occurred while training {model_name}: {e}")
        return None

# Model training and hyperparameter tuning
nb_params = {'alpha': [0.1, 0.2, 0.5, 1.0]}
best_nb = train_evaluate_model(MultinomialNB(), nb_params, X_train, y_train, X_test, y_test, "Naive Bayes")

svm_params = {'kernel': ['linear', 'sigmoid'], 'gamma': [0.1, 1.0], 'C': [0.1, 1.0]}
best_svm = train_evaluate_model(SVC(), svm_params, X_train, y_train, X_test, y_test, "SVM")

lr_params = {'C': [0.1, 1.0, 10.0]}
best_lr = train_evaluate_model(LogisticRegression(), lr_params, X_train, y_train, X_test, y_test, "Logistic Regression")

rf_params = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30]}
best_rf = train_evaluate_model(RandomForestClassifier(), rf_params, X_train, y_train, X_test, y_test, "Random Forest")

knn_params = {'n_neighbors': [3, 5, 11, 19], 'weights': ['uniform', 'distance']}
best_knn = train_evaluate_model(KNeighborsClassifier(), knn_params, X_train, y_train, X_test, y_test, "KNN")

# Save the best models and vectorizer if they are not None
if best_nb: joblib.dump(best_nb, 'best_naive_bayes.pkl')
if best_svm: joblib.dump(best_svm, 'best_svm.pkl')
if best_lr: joblib.dump(best_lr, 'best_logistic_regression.pkl')
if best_rf: joblib.dump(best_rf, 'best_random_forest.pkl')
if best_knn: joblib.dump(best_knn, 'best_knn.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Save the accuracies to a file
with open('model_accuracies.pkl', 'wb') as f:
    joblib.dump(model_accuracies, f)
    
print("Models and vectorizers saved successfully.")
