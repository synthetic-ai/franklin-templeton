import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from runner import *

# Set the Matplotlib backend to 'Agg'
plt.switch_backend('Agg')

from sklearn.decomposition import PCA

app = Flask(__name__)

# Placeholder for storing the synthetic data
synthetic_data_global = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    global synthetic_data_global
    
    dataset = request.form['dataset']
    model = request.form['model']
    
    logs = "\nGenerate Synthetic Data\n"
    
    [original_data, synthetic_data] = get_original_and_synthetic_data()
    synthetic_data_global = synthetic_data
    
    logs += "\nRunning PCA for analysis\n"
    # PCA plot
    pca_image_url = 'data:image/png;base64,' + pca_plot(original_data, synthetic_data)
    
    logs += "\n Running Stats \n"
    # Calculate statistics
    df = get_original_vs_synthetic_stats(original_data, synthetic_data)
    correlation_matrix = get_corr_mat(original_data, synthetic_data)

    df_html = df.to_html(classes="table table-bordered")
    correlation_matrix_html = correlation_matrix.to_html(classes="table table-bordered")
    
    logs += "\n Performing Koglomorov-Smirnov Test \n"
    # Perform Kolmogorov-Smirnov Test for each column
    ks_results = {}
    for col in original_data.columns:
        statistic, p_value = ks_2samp(original_data[col], synthetic_data[col])
        ks_results[col] = {"statistic": statistic, "p_value": p_value}

    # Print the results and interpret p-values
    for col, result in ks_results.items():
        print(f"KS Test for {col}:")
        print(f"  Statistic: {result['statistic']}")
        print(f"  P-value: {result['p_value']}")

        if result['p_value'] < 0.05:
            logs += f" \n Interpretation: No significant difference (p-value > 0.05). Synthetic data is statistically similar to the original data for {col}.\n"
        else:
            logs += f" \n Interpretation: Significant difference (p-value ≤ 0.05). Synthetic data is not statistically similar to the original data for {col}.\n"    
    
    logs += "Nice to Meet you! Hasta la Vista!"
    return jsonify({
        'logs': logs,
        'pca_image_url': pca_image_url,
        'original_synthetic_stats_table' : df_html,
        'correlation_matrix': correlation_matrix_html,
    })

@app.route('/download_synthetic_data')
def download_synthetic_data():
    global synthetic_data_global
    
    if synthetic_data_global is None:
        return "No synthetic data available. Generate data first.", 400
    
    output = BytesIO()
    synthetic_data_global.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='synthetic_data.csv')

if __name__ == '__main__':
    app.run(debug=True)
