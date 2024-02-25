

import json
import argparse
from itertools import chain
from functools import partial
#import warnings
#warnings.filter("ignore")
import torch
from transformers import AutoTokenizer,Trainer,TrainingArguments
from transformers import AutoModelForTokenClassification,DataCollatorForTokenClassification
import evaluate
from datasets import Dataset,features
import numpy as np

data=json.load(open("/kaggle/input/pii-detection-removal-from-educational-data/train.json"))

all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i,l in enumerate(all_labels)}
id2label = {v:k for k,v in label2id.items()}

target = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 
    'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 
    'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL'
]

print(id2label)

def tokenize(example,tokenizer,label2id,max_length):
    text=[]
    labels=[]
    for t,l  in zip(
        example["tokens"],example["provided_labels"]):
        text.append(t)
        labels.extend([l]*len(t))
        ws=True
        if ws:
            text.append(" ")
            labels.append("O")
            
        tokenized=tokenizer("".join(text),return_offsets_mapping=True,max_length=max_length)
        labels=np.array(labels)
        text="".join(text)
        token_labels=[]
        for idx,end_idx in tokenized.offset_mapping:
            if idx==0 and end_idx==0:
                token_labels.append(label2id["O"])
                continue
            if idx < len(text) and end_idx <= len(text):  # Check if the index is within bounds
                if text[idx].isspace():
                    idx += 1
                token_labels.append(label2id[labels[idx]])
            else:
                token_labels.append(label2id["O"])

        length=len(tokenized.input_ids)
        return {**tokenized,"labels":token_labels,"length":length}

tokenizer=AutoTokenizer.from_pretrained("roberta-base")
ds=Dataset.from_dict({
    "full_text":[x["full_text"]for x in data],
    "document":[str(x["document"])for x in data],
    "tokens":[x["tokens"]for x in data],
    #"training_whitespace":[x["trailing_whitespace"]for x in data],
    "provided_labels":[x["labels"]for x in data]
})
ds=ds.map(tokenize,fn_kwargs={"tokenizer":tokenizer,"label2id":label2id,"max_length":1024},num_proc=5)

x = ds[1:5]
for r,g in zip(x["tokens"], x["provided_labels"]):
    if g != "O":
        print((r,g))
print("*" * 100)

for ids, labels in zip(x["input_ids"], x["labels"]):
    for id, label in zip(ids, labels):
        if id2label[label] != "O":
            token = tokenizer.convert_ids_to_tokens(id)
            print((token, id2label[label]))

print("*" * 100)

from seqeval.metrics import recall_score,precision_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

def compute_metrics(p,all_labels):
    predictions,labels=p
    predictions=np.argmax(predictions,axis=2)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1_score = (1 + 5*5) * recall * precision / (5*5*precision + recall)
    
    results = {
        'recall': recall,
        'precision': precision,
        'f1': f1_score
    }
    return results

model =AutoModelForTokenClassification.from_pretrained(
    "roberta-base",
    num_labels=len(all_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True)
collator=DataCollatorForTokenClassification(tokenizer,pad_to_multiple_of=16)

args = TrainingArguments(
    output_dir="output", 
    fp16=False,
    learning_rate=1e-5,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=3,
    report_to="none",
    evaluation_strategy="no",
    do_eval=False,
    save_total_limit=1,
    logging_steps=20,
    lr_scheduler_type='cosine',
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_ratio=0.1,
    weight_decay=0.01
)

#from kaggle_secrets import UserSecretsClient
#user_secrets = UserSecretsClient()
#secret_value_0 = user_secrets.get_secret("kaggle")

import wandb
#wandb.login(key=secret_value_0)
wandb.init(project="PII",config=args)

artifact = wandb.Artifact("train_data", type="dataset")

# Convert data to DataFrame for easy table creation
df = pd.DataFrame(data)

# Convert DataFrame to WandB Table
table = wandb.Table(dataframe=df)

# Log the Table to the artifact
artifact.add(table, "train_data")

# Log the dictionaries and target labels
wandb.log({"label2id": label2id, "id2label": id2label, "target_labels": target})

# Log the artifact
wandb.log_artifact(artifact)
sweep_config = {
    "method": "random",  
    "metric": {"goal": "maximize", "name": "f1"},
    "parameters": {
        "learning_rate": {"min": 1e-6, "max": 1e-4},
        "per_device_train_batch_size": {"values": [4, 8, 16,32]},
        "gradient_accumulation_steps": {"values": [1, 2, 3,4,5]},
        "weight_decay": {"min": 0.0, "max": 0.1},
        "warmup_ratio": {"min": 0.0, "max": 0.2},
        #"num_epochs": {4,6,8,10}
    },
}

sweep_id=wandb.sweep(sweep=sweep_config,project="PII")

args.wandb_project="PII"
args.wandb_sweep_id=sweep_id

trainer=Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics,all_labels=all_labels))

%%time
trainer.train()

for epoch in range(args.num_train_epoch):
  predictions=model.predict(batch_input)
  predictions_df=pd.DataFrame(predictions,columns=["Token","Predicted_Label","True_Label"])
  wandb.log({
    "predictions_Table":wandb.Table(dataframe=predictions_df) 
  })

wandb.log({"Final_metrics":final_metrics,"Model_Artifact":wandb.Artifact("Roberta-PII",type="model")})

