from flask import render_template, request, send_file, jsonify, Flask
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("prediction.html")

@app.route("/train")
def train_route():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return "Training Completed Successfully."
    except Exception as e:
        raise CustomException(e, sys)

@app.route("/predict", methods = ['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_detail , result_df = prediction_pipeline.run_pipeline()

            total = len(result_df)
            safe_count = int((result_df['Result'] == 'safe').sum())
            phishing_count = int((result_df['Result'] == 'phishing').sum())

            previous_rows = result_df.head(50).reset_index()
            previous_rows.rename(columns={'index': 'row_no'}, inplace= True)
            previous_rows['row_no'] = previous_rows['row_no'] + 1

            return render_template(
                'results.html',
                total = total,
                safe_count = safe_count,
                phishing_count = phishing_count,
                safe_pct = round(safe_count/ total * 100, 1),
                phishing_pct = round(phishing_count/total *100, 1),
                rows = previous_rows.to_dict(orient='records'),
                showing_preview = total > 50
            )

        else:
            return render_template('prediction.html')

    except Exception as e:
        raise CustomException(e, sys)

@app.route("/download")
def download():
    try:
        from src.entity.artifact_entity import PredictionFileDetail
        detail = PredictionFileDetail()
        return send_file(detail.prediction_file_path, download_name= detail.prediction_file_name, as_attachment= True)
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port= port, debug= False)