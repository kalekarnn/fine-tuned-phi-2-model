# Phi-2 Fine-tuning with LoRA

This project demonstrates fine-tuning the Microsoft Phi-2 model using LoRA (Low-Rank Adaptation) on the OpenAssistant Conversations Dataset.

[Live Demo](https://huggingface.co/spaces/kalekarnn/fine-tuned-phi-2-model)

## Sample Chat

![chat](.chat.png)

## Project Structure
```
phi_2_qlora/
├── train.py        # Training script
├── app.py          # Inference application with Gradio interface
└── README.md       # This file
```

## Setup

1. Install required packages:
```bash
pip install transformers peft trl datasets torch gradio

## Training
The training script ( train.py ) performs the following:

- Loads and quantizes the Phi-2 base model
- Configures LoRA adapters for efficient fine-tuning
- Processes the OpenAssistant dataset
- Trains the model using SFTTrainer
- Saves the trained adapter weights
To start training:
```bash
python train.py
```

Training configurations:

- Batch size: 4
- Learning rate: 2e-4
- Training steps: 500
- LoRA rank: 16
- LoRA alpha: 32

## Inference
The inference script ( app.py ) provides a Gradio chat interface for interacting with the fine-tuned model:

- Loads the base Phi-2 model
- Applies the trained LoRA adapter
- Provides a user-friendly chat interface
To start the chat interface:

```bash
python app.py
```

## Model Details
Base Model: Microsoft Phi-2

- Size: 2.7B parameters
- Context Length: 2048 tokens
- Training Method: LoRA (Parameter Efficient Fine-Tuning)
- Dataset: OpenAssistant Conversations
## Results
The model was trained for 500 steps with a final training loss of approximately 2.4. The training logs show stable convergence throughout the process.

## Usage Notes
- The chat interface maintains conversation history
- Responses are generated with temperature 0.7 for balanced creativity
- Maximum response length is set to 512 tokens
- The model runs on CPU by default but can be configured for GPU
## Limitations
- Maximum context length is 2048 tokens
- Running on CPU may result in slower inference
- Model responses should be reviewed for accuracy
