import pandas as pd
import random
import numpy as np

random.seed(42)

# Step 1: Load feedback CSV and calculate avg score per mentor
df_feedback = pd.read_csv("synthetic_feedback.csv")

# Add score column
score_map = {"Positive": 5, "Neutral": 3, "Negative": 1}
df_feedback["score"] = df_feedback["sentiment"].map(score_map)

# Average feedback score per mentor
mentor_scores = df_feedback.groupby("mentor_id")["score"].mean().reset_index()
mentor_scores.columns = ["mentor_id", "feedback_score"]
mentor_scores["feedback_score"] = mentor_scores["feedback_score"].round(2)

# Step 2: Generate synthetic resume features per mentor
mentor_ids = mentor_scores["mentor_id"].tolist()

data = []
for mentor_id in mentor_ids:
    experience_years = random.randint(1, 15)
    skills_count     = random.randint(3, 20)
    projects_count   = random.randint(1, 25)
    feedback_score   = mentor_scores[mentor_scores["mentor_id"] == mentor_id]["feedback_score"].values[0]

    # Mentor rating formula, scaled out of 10 (was out of 5 before)
    mentor_rating = round(
        (
            (experience_years * 2) +
            (skills_count * 1.5) +
            (projects_count * 1.2) +
            (feedback_score * 10)
        ) / 10,
        2
    )

    # Keep rating between 1 and 10
    mentor_rating = max(1.0, min(mentor_rating, 10.0))

    data.append([
        experience_years,
        skills_count,
        projects_count,
        feedback_score,
        mentor_rating
    ])

# Step 3: Save final ML dataset
df_final = pd.DataFrame(
    data,
    columns=[
        "experience_years",
        "skills_count",
        "projects_count",
        "feedback_score",
        "mentor_rating"
    ]
)

df_final.to_csv("final_dataset.csv", index=False)

print(df_final.head(10))
print(f"\nTotal rows: {len(df_final)}")
print(f"\nMin rating: {df_final['mentor_rating'].min()}")
print(f"Max rating: {df_final['mentor_rating'].max()}")
