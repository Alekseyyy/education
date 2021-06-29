# A simple streamlit webapp to demonstrate the capabilities of ML through streamlit
# By Snehan Kekre
# Transcribed by Aleksey (c. May 2021, w/ some commentary re. the code)

# Working with a dataset regarding mushrooms, this project uses three machine learning models
# and three different evaluation metrics to show off the features of 

import streamlit as st # the focal point of this guided project.
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title("Binary classification webapp") # we can write a title 
    st.sidebar.title("Binary classification webapp") # and we can add a title onto a sidebar

    st.markdown("Are your mushrooms edible or poisonous? ") # we can write stuff down with markdown
    st.sidebar.markdown("Are your mushrooms edible or poisonous? ") # and we can write stuff down in the sidebar using markdown

    @st.cache(persist=True) # Python decorator to speed up the loading of datasets
    def load_data():
        data = pd.read_csv("./mushrooms.csv") # I couldn't upload this to my GitHub repo cos' the file is too big. Sorry :-(
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.type 
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ["edible", "poisonous"]

    st.sidebar.subheader("Choose some classifiers")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model hyperparameters")
        C = st.sidebar.number_input("C (Regularisation parameter)", 0.01, 10.0, step=0.01, key="C")
        kernel = st.sidebar.radio("kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernel coefficient)", ("scale", "auto"), key="gamma")

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("classify", key="classify"):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model hyperparameters")
        C = st.sidebar.number_input("C (Regularisation parameter)", 0.01, 10.0, step=0.01, key="C LR")
        max_iter = st.sidebar.slider("Maxiumum number of iterations", 100, 500, key="max_iter")

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("classify", key="classify"):
            st.subheader("Logistic Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == "Random Forest":
        st.sidebar.subheader("Model hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 500, step=10, key="n_estimators")
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key="max_depth")
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("classify", key="classify"):
            st.subheader("Random Forest results")
            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom dataset for classification")
        st.write(df)

if __name__ == '__main__':
    main()
