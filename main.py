
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier


WEB_FILE = "web/index.html"
FEATURES = ["studytime", "failures", "absences", "G1", "G2", "schoolsup", "famsup", "paid"]
BINARY_FEATURES = ["schoolsup", "famsup", "paid"]
PASS_MARK = 10

class StudentResultPredictor:
   
    def __init__(self):
        self.df = pd.read_csv("data/student_mat_uci_raw.csv")
        self.df["schoolsup"] = self.df["schoolsup"].map({"yes": 1, "no": 0})
        self.df["famsup"] = self.df["famsup"].map({"yes": 1, "no": 0})
        self.df["paid"] = self.df["paid"].map({"yes": 1, "no": 0})
        self.df["G3"] = (self.df["G3"] >= PASS_MARK).astype(int)
        
        self.X = self.df[FEATURES]
        self.y = self.df["G3"]
        self.model = None
        self.support_rates = {}
        

    def train(self):
        df = self.df
        X = self.X
        y = self.y
        print("Dataset shape", df.shape)
        print("Missing values:")
        print(df.isna().sum())
        print("Feature shape:", X.shape)
        print("Target shape:", y.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training feature shape:", X_train.shape)
        print("Test feature shape:", X_test.shape)
        print("Training target shape:", y_train.shape)
        print("Test target shape:", y_test.shape)

        default_model = DecisionTreeClassifier(random_state=42)
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        default_acc = accuracy_score(y_test, y_pred)
        print("Default Decision Tree accuracy:", round(default_acc, 4))
        print("Default confusion matrix:")
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

        grid_param = {
            "criterion": ["gini", "entropy"],
            "max_depth": [1, 2, 5, None],
            "min_samples_split": [2, 4, 12],
            "min_samples_leaf": [1, 2, 3],
        }
        grid_search = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
            param_grid=grid_param,
            cv=5,
            scoring="accuracy",
        )
        grid_search.fit(X_train, y_train)

        print("Best cross-validation accuracy:", round(grid_search.best_score_, 4))
        print("Best parameters:", grid_search.best_params_)

        tuned_model = grid_search.best_estimator_
        self.model = tuned_model
        for column in BINARY_FEATURES:
            self.support_rates[column] = {
                0: float(y[self.df[column] == 0].mean()),
                1: float(y[self.df[column] == 1].mean()),
            }
        y_pred_tuned = tuned_model.predict(X_test)
        tuned_acc = accuracy_score(y_test, y_pred_tuned)
        print("Tuned test accuracy:", round(tuned_acc, 4))
        print("Confusion matrix [needs support, pass]:")
        print(confusion_matrix(y_test, y_pred_tuned, labels=[0, 1]))
        print("Classification report:")
        print(classification_report(
            y_test, y_pred_tuned, labels=[0, 1],
            target_names=["Needs support", "Pass"], zero_division=0
        ))

    def predict(self, values: dict) -> dict:
        for column in BINARY_FEATURES:
            values[column] = 1 if values[column] == "yes" else 0
        row = pd.DataFrame([[values[column] for column in FEATURES]], columns=FEATURES)
        result = int(self.model.predict(row)[0])
        probabilities = self.model.predict_proba(row)[0]
        model_probability = float(probabilities[list(self.model.classes_).index(1)])
        support_probability = sum(
            self.support_rates[column][values[column]] for column in BINARY_FEATURES
        ) / len(BINARY_FEATURES)
        pass_probability = (model_probability * 0.8) + (support_probability * 0.2)
        support_summary = (
            f"School support: {'Yes' if values['schoolsup'] else 'No'}, "
            f"Family support: {'Yes' if values['famsup'] else 'No'}, "
            f"Paid classes: {'Yes' if values['paid'] else 'No'}"
        )
        return {
            "predicted_result": "Pass" if result else "Needs support",
            "pass_probability_pct": round(pass_probability * 100, 1),
            "support_summary": support_summary,
            "recommendation": (
                "Maintain a steady study routine and ask for help early if needed."
                if result else "Consider extra practice and speak with a teacher or academic adviser."
            ),
            "selected_model": "DecisionTreeClassifier",
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = StudentResultPredictor()
    app.state.predictor.train()
    yield


app = FastAPI(title="Exam Result Prediction", version="1.0.0", lifespan=lifespan)


class StudentInput(BaseModel):
    studytime: int = Field(ge=1, le=4, description="Study-time band, 1 to 4")
    failures: int = Field(ge=0, le=4)
    absences: int = Field(ge=0, le=93)
    G1: int = Field(ge=0, le=20, description="First-period grade")
    G2: int = Field(ge=0, le=20, description="Second-period grade")
    schoolsup: Literal["yes", "no"]
    famsup: Literal["yes", "no"]
    paid: Literal["yes", "no"]


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_FILE)


@app.get("/health")
def health():
    return {"status": "ok", "model_available": app.state.predictor.model is not None}


@app.get("/analysis")
def analysis():
    df = app.state.predictor.df
    passed = int((df["G3"] >= PASS_MARK).sum())
    count = len(df)
    return {
        "record_count": count,
        "average_final_grade": round(float(df["G3"].mean()), 2),
        "average_first_period_grade": round(float(df["G1"].mean()), 2),
        "average_second_period_grade": round(float(df["G2"].mean()), 2),
        "average_absences": round(float(df["absences"].mean()), 2),
        "pass_count": passed,
        "needs_support_count": count - passed,
        "pass_rate_pct": round(100 * passed / count, 1),
    }


@app.post("/predict")
def predict(student: StudentInput):
    return app.state.predictor.predict(student.model_dump())


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)

