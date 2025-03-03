# topicgen

It automatically generates relevant GitHub topics from **GitHub URL**. It uses zero-shot to analyze repository content and suggest appropriate topics based on the selected categories.

Try here : [Hugging Face Space](https://huggingface.co/spaces/Namgyu-Youn/topic-generator)


## Project Structure
### Project Overview
![project_architecture](https://github.com/user-attachments/assets/dbe27e6e-0a94-4151-8733-4f6e3c16b800)

### Data Collection Pipeline
![data_collection_pipeline](https://github.com/user-attachments/assets/868250d0-7309-483b-9b35-cdcae71a96a1)

### Sample Database
![Image](https://github.com/user-attachments/assets/f7155cc7-4f53-46a3-80d3-bc2a65153c4d)

### Model Training Pipeline
![training_pipeline](https://github.com/user-attachments/assets/d75ab603-fa3f-4d46-abc0-5c270571af23)

## ✨ Features
- **Collects GitHub repository data** (metadata, topics, READMEs) via GitHub API
- **Analyzes repository content** to predict relevant topics using ML models
- Trains a BERT-based **multi-label classifier** for topic prediction
- Stores repository and topic data in SQLite for efficient retrieval
- Exports trained models to ONNX format for production deployment

## 🚩 How to use?

```bash
git clone https://github.com/Namgyu-Youn/topicgen.git
cd topicgen
```

### Option 1: Using Poetry (Highly Recommended)
```bash
curl -sSL https://install.python-poetry.org | python3 - # Optional
poetry install

# Data Collection Pipeline
poetry run python -m topicgen.pipelines.data_collection_pipeline --min-stars 1000 --language python --max-repos 500

# Model Training Pipeline
poetry run python -m topicgen.pipelines.model_training_pipeline --base-model bert-base-uncased --num-epochs 5
```

### Option 2: Using Docker
```bash
# Build the Docker image
docker build -t github-topic-generator .

# Run data collection pipeline
docker run github-topic-generator python -m topicgen.pipelines.data_collection_pipeline

# Run model training pipeline
docker run github-topic-generator python -m topicgen.pipelines.model_training_pipeline
```

### Option 3: Standard Python Setup
```bash
python -m venv env

# On Windows
env\Scripts\activate
# On macOS/Linux
source env/bin/activate

pip install -r requirements.txt

# Data Collection Pipeline
python -m topicgen.pipelines.data_collection_pipeline

# Model Training Pipeline
python -m topicgen.pipelines.model_training_pipeline
```

## 🧐 Introduction about gradio UI

1. Enter GitHub URL
2. Select the main, sub category that best matches your repository
3. Click "Generate Topics" to get your results
4. Enjoy generated topics('#')! It can be used like this.


## 👥 Contribution guide : [CONTRIBUTING.md](https://github.com/Namgyu-Youn/github-topic-generator/blob/main/CONTRIBUTING.md)
Thanks for your interest. I always enjoy meaningful collaboration. <br/>
Do you have any question or bug?? Then submit **ISSUE**. You can also use awesome labels(🏷️).