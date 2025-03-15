'''
import tkinter as tk
import fraud_detection_model  # Import the fraud detection model

def predict_fraud():
    text_input = text_entry.get("1.0", tk.END).strip()
    if not text_input:
        result_text.set("⚠️ Please enter a message to analyze.")
        return

    prediction = modeltext.predict_fraud(text_input)  # Call fraud detection function
    result_text.set(prediction)

    # Clear the result after 5 seconds
    root.after(5000, lambda: result_text.set(""))

# Initialize GUI
root = tk.Tk()
root.title("Fraud SMS Detector")

tk.Label(root, text="Enter SMS Text:", font=("Arial", 12)).pack(pady=5)

text_entry = tk.Text(root, height=5, width=50)
text_entry.pack(pady=5)

tk.Button(root, text="Check for Fraud", command=predict_fraud).pack(pady=10)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 14), fg="red")
result_label.pack(pady=10)

root.mainloop()
'''


import tkinter as tk
from tkinter import messagebox
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = joblib.load("fraud_sms_rf.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_fraud():
    message = entry.get("1.0", "end-1c").strip()
    if not message:
        messagebox.showerror("Error", "Please enter a message.")
        return
    
    # Transform message using the loaded vectorizer
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    
    label_text.set("⚠️ Fraud" if prediction == 1 else "✅ Safe")
    label_result.config(fg="red" if prediction == 1 else "green")

# GUI Setup
root = tk.Tk()
root.title("Fraud SMS Detector")
root.geometry("400x300")

# Input field
entry_label = tk.Label(root, text="Enter SMS:")
entry_label.pack()
entry = tk.Text(root, height=5, width=50)
entry.pack()

# Predict button
predict_button = tk.Button(root, text="Check Fraud", command=predict_fraud)
predict_button.pack()

# Output label
label_text = tk.StringVar()
label_text.set("Prediction will appear here")
label_result = tk.Label(root, textvariable=label_text, font=("Arial", 12))
label_result.pack()

# Run GUI
root.mainloop()
