import joblib

model = joblib.load("models/nlp_tfidf_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("===================================")
print("       NLP ENGINE TEST")
print("===================================")

history = input("\nEnter transaction history:\n")

X = vectorizer.transform([history])

prediction = model.predict(X)[0]
probability = model.predict_proba(X)[0][1]

print("\nTransaction History:")
print(history)

print("\nNLP Risk Score:", round(probability, 4))

if prediction == 1:
    print("Prediction: HIGH RISK USER")
else:
    print("Prediction: NORMAL USER")