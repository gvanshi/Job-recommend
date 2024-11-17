from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Configure Matplotlib to use a non-GUI backend
matplotlib.use('Agg')

app = Flask(__name__)

# Load Data
data_dir = 'D:/projects/Project final year/project_name/data/raw/'
students_data = pd.read_excel(os.path.join(data_dir, 'students_data.xlsx'))
educators_data = pd.read_excel(os.path.join(data_dir, 'educators_data.xlsx'))
industry_data = pd.read_excel(os.path.join(data_dir, 'industry_data.xlsx'))
job_roadmap_data = pd.read_excel(os.path.join(data_dir, 'job_roadmap.xlsx'))

# Ensure static directory exists for plots
plot_dir = './static/plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Generate Plots
def generate_plots():
    try:
        # Students Data Analysis
        alignment = students_data['Do you feel that the technical courses in your curriculum are aligned with real-world industry needs?'].value_counts()
        alignment.plot(kind='bar', color='skyblue')
        plt.title("Curriculum Alignment with Industry Needs (Students)")
        plt.ylabel("Count")
        plt.xlabel("Responses")
        plt.savefig(os.path.join(plot_dir, 'students_alignment.png'))
        plt.close()

        # Programming Languages and Confidence
        languages = students_data['Which programming languages have you learned in your academic curriculum?'].value_counts()
        confidence = students_data['Do you feel confident using the programming languages for industry-relevant tasks?'].value_counts()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        languages.plot(kind='bar', ax=ax[0], color='green')
        ax[0].set_title("Programming Languages Learned")
        ax[0].set_ylabel("Count")
        ax[0].set_xlabel("Languages")
        confidence.plot(kind='bar', ax=ax[1], color='orange')
        ax[1].set_title("Confidence in Programming Languages")
        ax[1].set_ylabel("Count")
        ax[1].set_xlabel("Confidence Level")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'programming_confidence.png'))
        plt.close()

        # Educators Data Analysis
        methods = educators_data['Which teaching methods do you primarily use in your courses?'].value_counts()
        methods.plot(kind='bar', color='blue')
        plt.title("Preferred Teaching Methods (Educators)")
        plt.ylabel("Count")
        plt.xlabel("Methods")
        plt.savefig(os.path.join(plot_dir, 'educators_methods.png'))
        plt.close()

        # Industry Data Analysis - Graduate Preparedness
        preparedness = industry_data['How well-prepared are recent graduates for technical industry roles?'].value_counts()
        preparedness.plot(kind='bar', color='skyblue')
        plt.title("Graduate Preparedness for Industry")
        plt.ylabel("Count")
        plt.xlabel("Preparedness Level")
        plt.savefig(os.path.join(plot_dir, 'graduate_preparedness.png'))
        plt.close()

        # Technical Skills Expected from Graduates
        skills = industry_data['What are the most important technical skills that you expect graduates to have? (Select all that apply)'].value_counts()
        skills.plot(kind='bar', color='green')
        plt.title("Technical Skills Expected from Graduates")
        plt.ylabel("Count")
        plt.xlabel("Technical Skills")
        plt.savefig(os.path.join(plot_dir, 'technical_skills.png'))
        plt.close()

        # Key Knowledge Gaps in Graduates
        gaps = industry_data['What are the key knowledge gaps that you notice in recent graduates entering your industry? (Select all that apply)'].value_counts()
        gaps.plot(kind='bar', color='orange')
        plt.title("Key Knowledge Gaps in Graduates")
        plt.ylabel("Count")
        plt.xlabel("Knowledge Gaps")
        plt.savefig(os.path.join(plot_dir, 'key_knowledge_gaps.png'))
        plt.close()

        # Hands-on Training Value
        training_value = industry_data['How valuable is hands-on training (e.g., internships, co-op programs) in preparing students for industry roles?'].value_counts()
        training_value.plot(kind='bar', color='purple')
        plt.title("Value of Hands-on Training in Industry Preparation")
        plt.ylabel("Count")
        plt.xlabel("Value of Hands-on Training")
        plt.savefig(os.path.join(plot_dir, 'hands_on_training_value.png'))
        plt.close()

    except Exception as e:
        print(f"Error generating plots: {e}")

generate_plots()

# Train or Load AI Model for Recommendations
def train_or_load_model():
    model_path = './career_recommendation_model.pkl'
    encoder_path = './job_label_encoder.pkl'  # Path to save/load the label encoder

    # Check if model and encoder already exist
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        # Load model and label encoder if they exist
        model = joblib.load(model_path)
        job_label_encoder = joblib.load(encoder_path)
        return model, job_label_encoder

    # If they do not exist, train the model and label encoder
    data = job_roadmap_data.copy()

    # Initialize and fit the label encoder
    job_label_encoder = LabelEncoder()
    job_label_encoder.fit(data['Job Role'])

    data['Skills_Length'] = data['Required Skills'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)
    data['Certifications_Length'] = data['Recommended Certifications'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)
    data['Demand_Level_Encoded'] = LabelEncoder().fit_transform(data['Industry Demand Level'])

    X = data[['Skills_Length', 'Certifications_Length', 'Demand_Level_Encoded']]
    y = job_label_encoder.transform(data['Job Role'])  # Use the fitted label encoder here

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the model and label encoder
    joblib.dump(model, model_path)
    joblib.dump(job_label_encoder, encoder_path)  # Save the label encoder as well

    return model, job_label_encoder


recommendation_model, job_label_encoder = train_or_load_model()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/roadmaps')
def roadmaps():
    roadmap_data = job_roadmap_data.to_dict(orient='records')
    return render_template('roadmaps.html', roadmap_data=roadmap_data)

@app.route('/job/<job_role>')
def job_details(job_role):
    job_role = job_role.replace('_', ' ')
    job = job_roadmap_data[job_roadmap_data['Job Role'] == job_role].to_dict(orient='records')
    if not job:
        return render_template('404.html', message=f"Job Role '{job_role}' not found.")
    return render_template('job_details.html', job=job[0])

@app.route('/skills')
def skills():
    skills = students_data['Which programming languages have you learned in your academic curriculum?'].dropna()
    skills_series = skills.str.split(',').explode().value_counts()
    skill_labels = skills_series.index.tolist()
    skill_counts = skills_series.values.tolist()
    top_skills = skills_series.head(5).items()
    return render_template('skills.html', skill_labels=skill_labels, skill_counts=skill_counts, top_skills=top_skills)

@app.route('/visualizations')
def visualizations():
    plot_files = os.listdir(plot_dir)
    plots = [f'/static/plots/{plot}' for plot in plot_files if plot.endswith(('.png', '.jpg'))]
    return render_template('visualizations.html', plots=plots)

@app.route('/recommend', methods=['POST'])
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        skills = len(data['skills'].split(',')) if data['skills'] else 0
        certifications = len(data['certifications'].split(',')) if data['certifications'] else 0
        demand_level = int(data['demand_level'])
        
        # Prepare input for the model
        X_input = [[skills, certifications, demand_level]]

        # Make a prediction using the trained model
        prediction = recommendation_model.predict(X_input)

        # Handle unseen labels gracefully
        try:
            job = job_label_encoder.inverse_transform(prediction)[0]
        except ValueError as e:
            return jsonify({"error": f"Error: {e}. This job role is not recognized by the model."}), 400

        # Return the recommendation as a JSON response
        return jsonify({"recommended_job": job})

    except Exception as e:
        print(f"Error in recommendation: {e}")
        return jsonify({"error": "An error occurred during recommendation"}), 500



@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    recommended_job = data['recommended_job']
    user_feedback = data['feedback']
    with open("feedback_log.txt", "a") as log_file:
        log_file.write(f"Recommended: {recommended_job}, Feedback: {user_feedback}\n")
    return jsonify({"message": "Feedback recorded. Thank you!"})

if __name__ == '__main__':
    app.run(debug=True)
