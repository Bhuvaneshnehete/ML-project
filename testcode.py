import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tkinter as tk
from tkinter import ttk, messagebox

# --- Step 1: Load and Clean Data ---
df = pd.read_csv("adult.csv")
df.columns = df.columns.str.strip()  # Remove extra spaces from column names

# Replace '?' with NaN and drop missing
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Drop irrelevant columns
df.drop(['fnlwgt', 'native-country', 'race', 'capital-gain', 'capital-loss'], axis=1, inplace=True)

# Remove low education entries
low_edu = ['1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'Preschool']
df = df[~df['education'].isin(low_edu)]

# Remove noisy categories in workclass and occupation
df = df[~df['workclass'].isin(['?', 'Self-emp-not-inc', 'Without-pay'])]
df = df[df['occupation'] != '?']

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Map education → educational-num
edu_num_map = df[['education', 'educational-num']].drop_duplicates().set_index('education')['educational-num'].to_dict()

# --- Step 2: Balance Dataset by Undersampling ---
df_majority = df[df['income'] == 0]
df_minority = df[df['income'] == 1]
df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)
df = pd.concat([df_majority_downsampled, df_minority], axis=0).sample(frac=1, random_state=42)

# Features & Target
X = df.drop("income", axis=1)
y = df["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Train Models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC()
}

accuracies = {}
print("Model Evaluation:")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(f"{name}:\n  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}\n")
    accuracies[name] = acc

best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"✅ Best Model Selected: {best_model_name}")

# --- Step 4: GUI App ---
def predict():
    try:
        age = int(entry_age.get())
        gender = gender_var.get()
        occupation = occupation_var.get()
        workclass = workclass_var.get()
        hours = int(hours_var.get())
        education = education_var.get()

        for var, label in [
            (gender, "gender"),
            (occupation, "occupation"),
            (workclass, "workclass"),
            (education, "education")
        ]:
            if var not in label_encoders[label].classes_:
                messagebox.showerror("Invalid Input", f"Invalid {label}: {var}")
                return

        gender_encoded = label_encoders['gender'].transform([gender])[0]
        occupation_encoded = label_encoders['occupation'].transform([occupation])[0]
        workclass_encoded = label_encoders['workclass'].transform([workclass])[0]
        education_encoded = label_encoders['education'].transform([education])[0]
        edu_num = edu_num_map[education_encoded]

        input_data = pd.DataFrame([{
            'age': age,
            'workclass': workclass_encoded,
            'education': education_encoded,
            'educational-num': edu_num,
            'marital-status': label_encoders['marital-status'].transform(['Never-married'])[0],
            'occupation': occupation_encoded,
            'relationship': label_encoders['relationship'].transform(['Not-in-family'])[0],
            'gender': gender_encoded,
            'hours-per-week': hours
        }])

        prediction = best_model.predict(input_data)[0]
        result = label_encoders['income'].inverse_transform([prediction])[0]
        messagebox.showinfo("Prediction", f"Predicted Income: {result}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# --- GUI Layout ---
root = tk.Tk()
root.title("Income Predictor")

tk.Label(root, text="Age").grid(row=0, column=0, padx=10, pady=5)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Gender").grid(row=1, column=0, padx=10, pady=5)
gender_var = tk.StringVar()
gender_dropdown = ttk.Combobox(root, textvariable=gender_var, state="readonly")
gender_dropdown['values'] = list(label_encoders['gender'].classes_)
gender_dropdown.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Occupation").grid(row=2, column=0, padx=10, pady=5)
occupation_var = tk.StringVar()
occupation_dropdown = ttk.Combobox(root, textvariable=occupation_var, state="readonly")
occupation_dropdown['values'] = list(label_encoders['occupation'].classes_)
occupation_dropdown.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Workclass").grid(row=3, column=0, padx=10, pady=5)
workclass_var = tk.StringVar()
workclass_dropdown = ttk.Combobox(root, textvariable=workclass_var, state="readonly")
workclass_dropdown['values'] = list(label_encoders['workclass'].classes_)
workclass_dropdown.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Hours per Week").grid(row=4, column=0, padx=10, pady=5)
hours_var = tk.StringVar()
hours_dropdown = ttk.Combobox(root, textvariable=hours_var, state="readonly")
hours_dropdown['values'] = [20, 30, 35, 40, 45, 50, 60]
hours_dropdown.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Education").grid(row=5, column=0, padx=10, pady=5)
education_var = tk.StringVar()
education_dropdown = ttk.Combobox(root, textvariable=education_var, state="readonly")
education_dropdown['values'] = list(label_encoders['education'].classes_)
education_dropdown.grid(row=5, column=1, padx=10, pady=5)

tk.Button(root, text="Predict Income", command=predict, bg="lightblue").grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()
