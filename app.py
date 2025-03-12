import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

st.title("HR Analytics Employee Retention Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload HR Analytics CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    st.subheader("Employee Retention Count")
    left_count = df[df['left'] == 1].shape[0]
    retained_count = df[df['left'] == 0].shape[0]
    st.write(f"Employees Left: {left_count}")
    st.write(f"Employees Retained: {retained_count}")
    
    # Exploratory Data Analysis (EDA)
    st.subheader("Exploratory Data Analysis (EDA)")
    
    # Boxplots for key features
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y='satisfaction_level', x='left', ax=ax)
    plt.title("Satisfaction Level vs Employee Retention")
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y='last_evaluation', x='left', ax=ax)
    plt.title("Last Evaluation vs Employee Retention")
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y='average_montly_hours', x='left', ax=ax)
    plt.title("Average Monthly Hours vs Employee Retention")
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y='time_spend_company', x='left', ax=ax)
    plt.title("Time Spent in Company vs Employee Retention")
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y='number_project', x='left', ax=ax)
    plt.title("Number of Projects vs Employee Retention")
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    df_numeric = df.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df_numeric.corr(), linewidths=2, cmap="plasma", annot=True, ax=ax)
    st.pyplot(fig)
    
    # Employee Retention by Salary
    st.subheader("Employee Retention by Salary")
    df1 = df[['left', 'salary']]
    left = df1[df['left'] == 1].salary.value_counts()
    retained = df1[df['left'] == 0].salary.value_counts()
    counts = pd.DataFrame({"retained": retained, "left": left})
    fig, ax = plt.subplots()
    counts.plot(kind='bar', stacked=True, ax=ax, color=['blue', 'red'])
    plt.title("Retention grouped by Salary")
    plt.xlabel("Salary Level")
    plt.ylabel("Count")
    plt.legend(["Retained", "Left"])
    st.pyplot(fig)
    
    # Machine Learning Model
    st.subheader("Employee Attrition Prediction using Logistic Regression")
    df1 = df[['salary', 'Department', 'satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'left']]
    df1 = pd.get_dummies(df1, columns=['Department', 'salary'])
    X = df1.drop(columns=['left'])
    y = df1['left']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Model Evaluation
    st.subheader("Logistic Regression Model Analytics")
    st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.2f}")
    
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d", ax=ax)
    st.pyplot(fig)
    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(fig)