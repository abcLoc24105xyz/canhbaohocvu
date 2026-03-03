import pickle
import pandas as pd

model = pickle.load(open("academic_warning_model.pkl", "rb"))

test = pd.read_csv("test.csv")

test["combined_text"] = test["Advisor_Notes"].fillna("") + " " + test["Personal_Essay"].fillna("")

preds = model.predict(test)

submission = pd.DataFrame({
    "Student_ID": test["Student_ID"],
    "Academic_Status": preds
})

submission.to_csv("submission.csv", index=False)
print("Submission file created.")