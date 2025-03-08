# Fine-Tuning GPT-2 for Medical Query Response Generation

![Python: 3.7+](https://img.shields.io/badge/python-3.7%2B-blue)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%F0%9F%A4%A9-yellow)

##  Overview
This project fine-tunes a **GPT-2 Transformer model** to generate **contextually relevant responses to medical queries** based on subset of dataset [Malikeh1375/medical-question-answering-datasets](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets). The model is trained using **Parameter-Efficient Fine-Tuning (PEFT) with QLoRA**, reducing computational overhead while achieving high accuracy. 

The repository includes:
- **Training scripts** for fine-tuning GPT-2 on medical Q&A datasets.
- **Evaluation methods** to assess model performance.
- **Inference pipeline** for generating real-time responses.

---
##  Table of Contents  
- [Features](#features)
- [Fine-Tuning Process](#fine-tuning-process)
- [Optimized Generation Parameters](#optimized-generation-parameters)  
- [Dataset](#dataset)  
- [Installation & Setup](#installation-and-setup)  
- [Usage Example](#usage-example)
- [Project Structure](#project-structure)
---
##  Features  

- **GPT-2 Fine-Tuning**: Specifically fine-tuned GPT-2 model on medical query-response data.  
- **Parameter-Efficient Fine-Tuning (PEFT)**: Efficient training with reduced computational resources.  
- **Optimized Text Generation**: Improved response quality using advanced sampling strategies.  
- **Real-time Inference Pipeline**: Generate coherent medical responses instantly.  
- **Easy-to-use**: Simple setup and inference scripts provided.  

---
##  Fine-Tuning Process
1. **Dataset Preparation**  
   - Tokenization using Hugging Face `Tokenizer`
   - Preprocessing text to align with GPT-2 input format

2. **Model Fine-Tuning**  
   - Utilizing **QLoRA (Quantized LoRA) for PEFT**
   - Hyperparameter tuning (`learning_rate`, `batch_size`, `epochs`)

3. **Optimization & Sampling Techniques**  
   - **Temperature Scaling** (`temperature=0.7`)
   - **Top-k and Top-p Filtering** (`top_k=50`, `top_p=0.9`)
   - **Repetition Penalty** (`1.2`) to avoid redundant outputs

---
##  Optimized Generation Parameters  

The following generation parameters were optimized for high-quality outputs:  

```python
temperature = 0.7  
top_k = 50  
top_p = 0.9  
repetition_penalty = 1.2  
do_sample = True  
max_length = 200  
```

---
##  Dataset  

The fine-tuning process utilizes the subset of [Malikeh1375/medical-question-answering-datasets](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets) from Hugging Face Datasets. This comprises a wide range of medical question-answer pairs, making it ideal for training GPT-2 to generate contextually relevant responses in the healthcare domain.

---
## Installation and Setup 

### Step 1: Clone the Repository  
```bash
git clone https://github.com/vedants254/Finetuned-GPT2.git
cd Finetuned-GPT2
```

### Step 2: Create a virtual environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install torch transformers peft datasets accelerate bitsandbytes 
```

---
## Usage Example  

Run inference easily with the provided script or notebook:  

### Example Python Script:  
```python
from transformers import pipeline

generator = pipeline('text-generation', model="./gpt2_finetuned_medical")

prompt = "What are common symptoms of vitamin D deficiency?"
response = generator(
    prompt,
    max_length=200,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True
)

print(response[0]['generated_text'])
```


