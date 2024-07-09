import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv(r"C:\Users\harsh\Downloads\500_Person_Gender_Height_Weight_Index.csv")

# Preprocess the data, Handle missing values, etc.
# For gender, we will use Label Encoding
gender_encoder = LabelEncoder()
df['gender_encoded'] = gender_encoder.fit_transform(df['Gender'])

# Define features (X) and target (y)
X = df[['gender_encoded', 'Height', 'Weight']]
y = df['Index']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

def predict_bmi(gender, height, weight):
    try:
        
        height = float(height_entry.get()) 
        weight = float(weight_entry.get())

        # Convert gender to encoded value
        gender_encoded = gender_encoder.transform([gender])[0]

        # Predict BMI for the new data point
        new_data = pd.DataFrame({'gender_encoded': [gender_encoded], 'Height': [height], 'Weight': [weight]})
        predicted_bmi = model.predict(new_data)

        return predicted_bmi
    except ValueError:
        messagebox.showerror("Error", "Invalid input")

# Create the GUI
root = tk.Tk()
root.title("BMI Prediction")

root.iconbitmap(r"C:\Users\harsh\Downloads\bmi-icon.ico")

root.configure(bg="lightblue")

root.minsize(300,300)

root.maxsize(600,600)

gender_label = tk.Label(root, text="Gender:", font=("Arial",20,"bold"),bg="lightblue")
gender_label.pack()

gender_var = tk.StringVar()
gender_options = ['Male', 'Female']
gender_dropdown = tk.OptionMenu(root, gender_var, *gender_options)
gender_dropdown.configure(fg="black")
gender_dropdown.pack()

height_label = tk.Label(root, text="Height in cm:", font=("Arial",20,"bold"),bg="lightblue",borderwidth=3)
height_label.pack()
height_entry = tk.Entry(root)
height_entry.pack()

weight_label = tk.Label(root, text="Weight in kg:", font=("Arial",20,"bold"),bg="lightblue")
weight_label.pack()
weight_entry = tk.Entry(root)
weight_entry.pack()

def classify_bmi(predicted_bmi):
    if predicted_bmi < 18.5:
        return "Oops! You are Underweight\nTry to visit a doctor and follow a healthy eating pattern rich in vegetables,fruits and whole grains."
    elif 18.5 <= predicted_bmi < 24.9:
        return "Your BMI is within the Normal Weight range\nKeep up the healthy habits!"
    elif 25.0 <= predicted_bmi < 29.9:
        return "You are Pre-obese\nIt's a good time to focus on maintaining a balanced diet and engaging in regular physical activity to stay healthy."
    elif 30.0 <= predicted_bmi < 34.9:
        return "You are in Obesity Class I (Moderate obesity)\nConsider seeking professional guidance to adopt a healthier lifestyle."
    elif 35.0 <= predicted_bmi < 39.9:
        return "You are in Obesity Class II (Severe obesity)\nIt's crucial to prioritize your health and work towards significant lifestyle changes."
    else:
        return "You are in Obesity Class III (Very severe obesity)\nConsult a healthcare professional for personalized advice and support."

def on_predict_button_click():
    gender = gender_var.get()
    height = float(height_entry.get()) 
    weight = float(weight_entry.get())
    predicted_bmi = predict_bmi(gender, height, weight)
    classification = classify_bmi(predicted_bmi)

    result_label.config(text=f"Predicted BMI: {predicted_bmi[0]:.2f}\nClassification: {classification}")

predict_button = tk.Button(root, text="Predict BMI", command=on_predict_button_click,font=("Arial",20,"bold"),activebackground="pink")
predict_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
