# Psychology Model Fine-tuning Project

This project provides a basic structure for fine-tuning a Language Model for psychology-related tasks.

## Project Structure

```
psychology_model/
├── data/
│   └── dataset.py       # Dataset loading and preprocessing
├── models/
│   └── model.py         # Model definition
├── scripts/
│   ├── train.py         # Training script
│   └── evaluate.py      # Evaluation script
├── README.md            # This file
├── requirements.txt     # Project dependencies
├── Dockerfile           # Docker configuration file
└── .dockerignore        # Specifies intentionally untracked files that Docker should ignore
```

## Requirements

- Python 3.12
- PyTorch
- Transformers

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Quick Start Guide

1.  **Prepare your data:** Modify `data/dataset.py` to load and preprocess your psychology-related text data.
2.  **Configure training:** Edit `scripts/train.py` to set the model name, training data, and hyperparameters.
3.  **Run training:** Execute the training script: `python scripts/train.py`.
4.  **Evaluate your model:** After training, use `scripts/evaluate.py` to assess the model's performance. Modify the script to point to your fine-tuned model.

## Docker Instructions

1.  **Build the Docker image:**

    ```bash
    docker build -t psychology_model .
    ```

2.  **Run the Docker container:**

    ```bash
    docker run -it --rm -v $(pwd):/app psychology_model bash
    ```

    This command mounts the current directory (`$(pwd)`) into the container at `/app`. Any changes you make in the container will be reflected on your host machine, and vice versa.

3.  **Run training inside the container:**
    ```bash
    python scripts/train.py
    ```

## Note

Remember to adjust the `CMD` in the `Dockerfile` or the command you use to run the container to match the specific script you want to execute.
