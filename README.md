# FairVision - Age Group Classification

## 📌 Project Overview

FairVision is an AI-powered image classification application that predicts the age group of a person from a face image.

The application allows users to upload a face image and uses a Convolutional Neural Network (CNN) model to predict the most likely age group. The system displays the top three predictions together with their confidence scores.

> This project is developed for demonstration and educational purposes only.

## ✨ Features

- Upload a face image
- Predict the person's age group
- Display the top 3 predictions
- Show confidence scores for predictions
- Simple and user-friendly web interface
- Automatic model loading and downloading

## 🛠️ Technologies Used

- Python
- Streamlit
- PyTorch
- Torchvision
- Pillow
- NumPy
- gdown
- Convolutional Neural Network (CNN)

## 📋 Age Groups

The model classifies images into the following age groups:

- 0–2
- 3–9
- 10–19
- 20–29
- 30–39
- 40–49
- 50–59
- 60–69
- 70+

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/sirimanna2002/fairvision-app.git
cd fairvision-app
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

```bash Windows:
venv\Scripts\activate
```

```bash macOS / Linux:
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Running the Application

Start the application using:

```bash
streamlit run app.py
```

The application will open in your web browser.

## 📁 Project Structure

```text
fairvision-app/
│
├── app.py
├── requirements.txt
└── README.md
```

### ⚠️ Disclaimer

This project is developed for educational and demonstration purposes.
The predicted age group should not be considered an accurate measurement of a person's actual age.

### 👩‍💻 Author

## 👩‍💻 Author

**Malsha Nethmini**

🔗 **LinkedIn:** [Malsha Nethmini](https://www.linkedin.com/in/malsha-nethmini-vk/)
