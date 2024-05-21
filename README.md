# Fine-Tuning-BERT-for-Sentiment-Analysis-with-NeMo

## Project Overview <br>
This project demonstrates the process of fine-tuning a BERT model using NVIDIA's NeMo framework for a sentiment analysis task. The goal is to classify sentences from the Stanford Sentiment Treebank (SST-2) dataset into positive or negative sentiments.

### Setup and Installation
1. Setup NeMo: Ensure NeMo is installed and GPU is enabled.

```python
import nemo
from nemo.collections import nlp as nemo_nlp
from nemo.utils.exp_manager import exp_manager
import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
````

2. Prepare Dataset: Download and preprocess the SST-2 dataset.
```python
!wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
!unzip -o SST-2.zip -d {DATA_DIR}
!sed 1d {DATA_DIR}/SST-2/train.tsv > {DATA_DIR}/SST-2/train_nemo_format.tsv
!sed 1d {DATA_DIR}/SST-2/dev.tsv > {DATA_DIR}/SST-2/dev_nemo_format.tsv
```

### Model Configuration
Configure the NeMo model for sentiment analysis.

```python
config.model.dataset.num_classes = 2
config.model.train_ds.file_path = os.path.join(DATA_DIR, 'SST-2/train_nemo_format.tsv')
config.model.validation_ds.file_path = os.path.join(DATA_DIR, 'SST-2/dev_nemo_format.tsv')
config.trainer.max_epochs = 2
```

### Training
Train the model using the prepared dataset and configurations.

```python
trainer = pl.Trainer(**config.trainer)
model = nemo_nlp.models.TextClassificationModel(cfg=config.model, trainer=trainer)
trainer.fit(model)
```

### Evaluation
Evaluate the model's performance on the validation set

```python
eval_model = nemo_nlp.models.TextClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
eval_trainer.test(model=eval_model, verbose=False)
```

### Inference
Use the trained model for inference on new sentences.

```python
queries = ['Example sentence 1', 'Example sentence 2']
results = infer_model.classifytext(queries=queries, batch_size=3, max_seq_length=512)
for query, result in zip(queries, results):
    print(f'Query: {query}\nPredicted label: {result}')
```

### Conclusion
This project showcases the capabilities of NeMo for fine-tuning large language models for specific NLP tasks. The use of GPU acceleration and efficient data handling ensures high performance and scalability.


