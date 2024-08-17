from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import pandas as pd
import json
from ocr_functions import preprocess_image, extract_information, process_receipts
import plotly.graph_objects as go
import plotly.express as px

app = Flask(__name__)
app.secret_key = 'kartik'
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def df_to_json(df):
    return df.to_json(orient='split')

def json_to_df(json_str):
    return pd.read_json(json_str, orient='split')

@app.before_request
def initialize_session():
    if 'receipt_data' not in session:
        session['receipt_data'] = df_to_json(pd.DataFrame(columns=[
            'Invoice Number', 'Customer Name', 'Bill Date', 'Total Amount', 
            'Transaction Type', 'Payment Type', 'Currency', 'Comment'
        ]))

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('files')
    receipt_data = json_to_df(session['receipt_data'])

    transaction_type = request.form.get('transaction_type')
    payment_type = request.form.get('payment_type')
    currency = request.form.get('currency')
    comment = request.form.get('comment')

    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            new_data = process_receipts([file_path])  # Assuming this returns a DataFrame
            new_data['Transaction Type'] = transaction_type
            new_data['Payment Type'] = payment_type
            new_data['Currency'] = currency
            new_data['Comment'] = comment
            receipt_data = pd.concat([receipt_data, new_data], ignore_index=True)
    
    session['receipt_data'] = df_to_json(receipt_data)
    return redirect(url_for('main'))

@app.route('/report')
def report():
    receipt_data = json_to_df(session.get('receipt_data', df_to_json(pd.DataFrame())))
    bar_fig = px.bar(receipt_data, x='Transaction Type', y='Total Amount', title='Total Amount Spent per Transaction Type')
    bar_chart = bar_fig.to_html(full_html=False)

    # Pie Chart: Distribution of Expenses by Payment Type
    pie_fig = px.pie(receipt_data, names='Payment Type', values='Total Amount', title='Distribution of Expenses by Payment Type')
    pie_chart = pie_fig.to_html(full_html=False)

    # Line Chart: Expenses Over Time
    line_fig = px.line(receipt_data, x='Bill Date', y='Total Amount', title='Expenses Over Time')
    line_chart = line_fig.to_html(full_html=False)

    return render_template('report.html', tables=[receipt_data.to_html(classes='data', header="true")],
                           bar_chart=bar_chart, pie_chart=pie_chart, line_chart=line_chart)


if __name__ == "__main__":
    app.run(debug=True)
