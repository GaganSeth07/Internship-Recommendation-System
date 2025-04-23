# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# --- Load and Preprocess Data ---
df = pd.read_csv('Internships.csv')

# Rename columns
df = df.rename(columns={
    'Domain ': 'Domain',
    'Company Name': 'Company',
    'Duration (in months)': 'Duration_Days',
    'Stipend received(YES/NO)': 'Stipend_Received',
    'Stipend Amount(If YES)': 'Stipend_Amount'
})

# Clean whitespace in 'Domain' column
df['Domain'] = df['Domain'].str.strip()

# Encode stipend received
df['Stipend_Received'] = df['Stipend_Received'].map({'YES': 1, 'NO': 0})

# Fill missing stipend amount with 0
df['Stipend_Amount'] = df['Stipend_Amount'].fillna(0)

# Encode domain and company
domain_encoder = LabelEncoder()
df['Domain_Encoded'] = domain_encoder.fit_transform(df['Domain'])

company_encoder = LabelEncoder()
df['Company_Encoded'] = company_encoder.fit_transform(df['Company'])

# Reverse lookup for company decoding
company_decoder = dict(zip(df['Company_Encoded'], df['Company']))

# --- Train Decision Tree Model ---
X = df[['Domain_Encoded', 'Duration_Days']]
y = df['Company_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# --- Streamlit Web App ---

st.title("Internship Recommendation System")
st.write("Choose a domain to get the best internships !!")

# --- Dropdown for Domain Selection (Dynamic) ---

# Automatically get unique domains, sorted
domain_list = sorted(df['Domain'].unique())
domain_list.insert(0,"")  # Insert first blank option

user_domain = st.selectbox("Select Domain:", domain_list)

if st.button("Get Recommendations"):

    if user_domain == "":
        st.warning("⚠️ Please select a domain before proceeding!")
    else:
        st.success(f"✅ You selected: {user_domain}")

        # --- Rule-Based Recommendation ---
        st.subheader("📋 Top Recommendations:")

        filtered_df = df[df['Domain'] == user_domain]

        if not filtered_df.empty:
            stipend_internships = filtered_df[filtered_df['Stipend_Received'] == 1]
            non_stipend_internships = filtered_df[filtered_df['Stipend_Received'] == 0]

            stipend_internships = stipend_internships.sort_values(by='Stipend_Amount', ascending=False)
            non_stipend_internships = non_stipend_internships.sort_values(by='Duration_Days', ascending=False)

            combined_internships = pd.concat([stipend_internships, non_stipend_internships])

            top_3 = combined_internships.head(3).reset_index(drop=True)

            for i, row in top_3.iterrows():
                st.markdown(f"### #{i+1}")
                st.write(f"**Domain:** {row['Domain']}")
                st.write(f"**Company:** {row['Company']}")
                if row['Stipend_Received'] == 1:
                    st.write(f"**Stipend Amount:** ₹{int(row['Stipend_Amount'])}")
                st.write(f"**Duration (Days):** {int(row['Duration_Days'])}")
                st.markdown("---")
        else:
            st.error("❌ No internships available for the selected domain.")