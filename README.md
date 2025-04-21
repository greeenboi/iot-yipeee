# Cough Health Analyzer

A Gradio web application for analyzing cough audio to classify health status as 'healthy', 'COVID-19', or 'symptomatic'.

## Features

- **Upload Audio**: Upload audio files containing coughs for analysis
- **Record Audio**: Record coughs directly in the browser for analysis
- **Live Cough Detection**: Stream audio in real-time to detect and analyze coughs
- **API Access**: Make API calls to the model for integration with other applications

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required dependencies:
```bash
pip install -r requirements-gradio.txt
```

3. Run the application:
```bash
python app.py
```

The application will be available at http://localhost:7860 by default.

## Deploying to Hugging Face Spaces

To deploy this application to Hugging Face Spaces:

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Choose "Gradio" as the SDK
3. Upload the following files to your Space:
   - `app.py`
   - `cough_classification_model.pkl`
   - `requirements-gradio.txt` (rename to `requirements.txt`)

The Space will automatically build and deploy your application.

## Using the API

You can access the model via API calls. Here's an example using Python:

```python
import requests
import json

# For file upload
files = {'audio': open('your_audio_file.wav', 'rb')}
response = requests.post('YOUR_GRADIO_URL/api/predict', files=files)
result = json.loads(response.content)
print(result)
```

Replace `YOUR_GRADIO_URL` with the URL of your deployed Gradio app.

## Model Information

The model is a Random Forest classifier trained on audio features extracted from cough recordings. It classifies coughs into three categories:

- **Healthy**: Normal coughs from healthy individuals
- **COVID-19**: Coughs from individuals with COVID-19
- **Symptomatic**: Coughs from individuals with respiratory symptoms but not COVID-19

## Live Audio Streaming and Cough Detection

The application includes a feature for streaming live audio and detecting coughs in real-time. When a cough is detected, it is automatically analyzed by the model.

This feature is useful for:
- Continuous health monitoring
- Automated cough counting and analysis
- Real-time health status updates

## Limitations

- The model's accuracy depends on the quality of the audio recording
- Background noise can affect cough detection and classification
- The model should not be used as a substitute for professional medical diagnosis

## License

[Specify your license here]