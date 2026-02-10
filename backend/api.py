from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd

from pipeline.full_pipeline import run_complete_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all origins (dev only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_leaderboard(results_df, task):
    if task == "regression":
        metric = "Test_R2"
        best_idx = results_df[metric].idxmax()
    else:
        metric = "Test_Accuracy"
        best_idx = results_df[metric].idxmax()

    best_row = results_df.loc[best_idx]

    return {
        "model": best_row["Model"],
        "metric": metric,
        "value": float(best_row[metric])
    }


def plot_to_base64(fig):
    """
    Convert a matplotlib figure to base64 string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.post("/run-pipeline/")
async def run_pipeline_api(
    file: UploadFile = File(...),
    target: str = Form(...),
    task: str = Form(...)
):
    # -------------------------------
    # Save uploaded file
    # -------------------------------
    file_path = f"output/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # -------------------------------
    # Run ML pipeline
    # IMPORTANT: pipeline should return a DataFrame
    # -------------------------------
    print(f"filepath= {file_path}")
    df = pd.read_csv(file_path)
    results_df = run_complete_pipeline(
        df=df,
        target=target,
        task=task
    )

    # -------------------------------
    # Create matplotlib plot
    # -------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    if task == "regression":
        metrics = ["Train_R2", "Test_R2"]
    else:
        metrics = ["Train_Accuracy", "Test_Accuracy"]

    results_df.set_index("Model")[metrics].plot.bar(ax=ax)
    ax.set_title(f"Model Comparison ({task.capitalize()})")
    ax.set_ylabel("Score")
    ax.legend()

    # -------------------------------
    # Convert plot to base64
    # -------------------------------
    plot_base64 = plot_to_base64(fig)

    # -------------------------------
    # Return JSON response
    # -------------------------------
    leaderboard = get_leaderboard(results_df, task)

    return {
        "message": "Pipeline executed successfully",
        "leaderboard": leaderboard,
        "results": results_df.to_dict(),
        "plots": {
            "model_comparison": plot_base64
        }
    }

