import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import pandas as pd
from PIL import Image
from datetime import datetime

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
    image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    return image

def extract_information(text):
    invoice_number = re.search(r"INVOICE NO.*?\b([M][A-Z0-9|]{14,16})\b.*?\b([M][A-Z0-9|]{14,16})?\b", text)
    bill_date = re.search(r'(?:Date|Bill Date|Order Date|Closed Bi|Purchase Date|)(?:\s*[:])?\s*(\d{2}/\d{2}/\d{2,4})', text)
    customer_name = re.search(r"CUSTOMER\s+NAME\s+(?:CUSTOMER\s+GSTIN\s+)?([A-Za-z]+\s+[A-Za-z]+)", text, re.IGNORECASE)
    total_amount = re.search(r"(?:Total Booking(?: Amount)?|Grand Total)\.?\s*(?:INR|Inr)\s*([\d,]+\.\d{1,2})", text)

    return {
        'Invoice Number': invoice_number.group(1) if invoice_number else None,
        'Customer Name': customer_name.group(1) if customer_name else None,
        'Bill Date': bill_date.group(1) if bill_date else None,
        'Total Amount': total_amount.group(1) if total_amount else None
    }

def convert_date_format(date_str):
    formats_to_try = ['%d/%m/%Y', '%m-%d-%Y', '%m/%d/%y', '%d-%m-%y', '%m/%d/%Y', '%d-%m-%Y']
    for fmt in formats_to_try:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.strftime('%d/%m/%Y')
        except ValueError:
            continue
    return date_str

def process_receipts(receipt_images):
    data = []
    for image_path in receipt_images:
        preprocessed_image = preprocess_image(image_path)
        result = ocr.ocr(preprocessed_image, cls=True)
        text = [line[1][0] for line in result[0]]
        text = ' '.join(text)
        extracted_data = extract_information(text)
        data.append(extracted_data)
    df = pd.DataFrame(data)
    #df['Bill Date'] = df['Bill Date'].apply(convert_date_format)
    #df['Bill Date'] = pd.to_datetime(df['Bill Date'], format='%d/%m/%Y')
    return df
