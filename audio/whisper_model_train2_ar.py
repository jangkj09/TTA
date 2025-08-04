import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import soundfile as sf
import io
import pyarrow.parquet as pq
import torchaudio
import evaluate
# Configuration
import librosa
from sklearn.model_selection import train_test_split


import sys

lr=float(sys.argv[1]) if len(sys.argv) > 1 else 0.0001

class Config:
    parent_dir = "/home/jangkj/gitRepo/TTA_data/AR/"  # Parent directory containing parquet files
    # parquet_dir = "/data1/wuyinjun/datasets/librispeech/clean/train.100/"  # Directory containing parquet files
    sample_rate = 16000  # Whisper expects 16kHz audio
    batch_size = 8  # Mini-batch size
    learning_rate = lr
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "openai/whisper-small"
    save_dir = "/home/jangkj/gitRepo/TTA/finetune_models/whisper_ar/"
    max_audio_length = 30 * sample_rate  # 30 seconds max

# Create dataset class that reads directly from parquet
class ParquetAudioDataset(Dataset):
    def __init__(self, parquet_paths, processor):
        self.parquet_paths = parquet_paths
        self.processor = processor
        # self.file_handles = [pq.ParquetFile(path) for path in parquet_paths]
        # self.df_ls = [pd.read_parquet(path) for path in parquet_paths]
        # self.row_counts = [file.metadata.num_rows for file in self.file_handles]
        # self.cumulative_rows = np.cumsum([0] + self.row_counts)
        # self.total_rows = sum(self.row_counts)
        
    def __len__(self):
        return len(self.parquet_paths)
    
    def __getitem__(self, idx):
        # Determine which file contains this index
        # file_idx = np.searchsorted(self.cumulative_rows, idx, side='right') - 1
        # row_idx = idx - self.cumulative_rows[file_idx]
        # # table = self.file_handles[file_idx].read_row_group(0, columns=['audio', 'text'])
        # # batch = table.slice(row_idx, 1).to_pandas()
        # item = self.df_ls[file_idx].iloc[row_idx]
        # audio_data = item["audio_data"]
        audio_data, sample_rate = librosa.load(self.parquet_paths[idx] + ".wav", sr=None)
        text = ""
        with open(self.parquet_paths[idx] + ".txt", "r") as f:
            text += f.read().strip()
        # audio_data, sample_rate = sf.read(io.BytesIO(item['audio']["bytes"]))
        # audio_data=torchaudio.transforms.Resample(audio_data, new_freq=sample_rate/10)
        
        # Get audio data (assuming it's stored as numpy array or bytes)
        # audio_data = batch['audio'].iloc[0]
        
        # Handle different audio storage formats
        # if isinstance(audio_data, bytes):
        #     # If audio is stored as bytes (e.g., WAV file bytes)
        #     audio, _ = sf.read(io.BytesIO(audio_data))
        # elif isinstance(audio_data, np.ndarray):
        #     # If audio is stored directly as numpy array
        #     audio = audio_data
        # else:
        #     raise ValueError(f"Unsupported audio format: {type(audio_data)}")
        
        # # Convert to mono if needed
        # if len(audio.shape) > 1:
        #     audio = np.mean(audio, axis=0)
            
        # # Normalize audio
        # audio = audio / np.max(np.abs(audio))
        
        # Get transcription
        # text = item['text']
        
        return {
            'audio': audio_data,
            'text': text
        }

# Collate function for DataLoader
def collate_fn(batch):
    processor = Config.processor
    
    # Process audio
    audio_arrays = [item['audio'] for item in batch]
    input_features = processor(
        audio_arrays, 
        sampling_rate=Config.sample_rate, 
        return_tensors="pt", 
        padding="max_length",
        truncation=True
        # padding=True
    ).input_features
    
    # Process labels
    texts = [item['text'].lower() for item in batch]
    labels = processor(text=texts, return_tensors="pt", padding=True).input_ids
    
    return {
        'input_features': input_features,
        'labels': labels,
        'texts': texts
    }

# Initialize processor and model
processor = WhisperProcessor.from_pretrained(Config.model_name)
model = WhisperForConditionalGeneration.from_pretrained(Config.model_name).to(Config.device)

# Set processor in Config for collate_fn access
Config.processor = processor

# Find all parquet files in directory

all_sample_names = [f.split(".txt")[0] for f in os.listdir(Config.parent_dir) if f.endswith('.txt')]
sample_count = len(all_sample_names)
print(f"Total samples found: {sample_count}")

train_sample_ids, test_sample_ids = train_test_split(
    all_sample_names, test_size=0.2, random_state=42
)
# random_sample_ids = np.random.permutation(sample_count)
# train_sample_ids = random_sample_ids[:int(0.8 * sample_count)]
# test_sample_ids = random_sample_ids[int(0.8 * sample_count):]


train_sample_names = [os.path.join(Config.parent_dir,name) for name in train_sample_ids]
test_sample_names = [os.path.join(Config.parent_dir, name) for name in test_sample_ids]

# train_sample_names = [os.path.join(Config.parent_dir,all_sample_names[i]) for i in train_sample_ids]
# test_sample_names = [os.path.join(Config.parent_dir, all_sample_names[i]) for i in test_sample_ids]


# train_dirs = [os.path.join(Config.parent_dir, f) 
#                  for f in os.listdir(Config.parent_dir) 
#                  if os.path.isdir(os.path.join(Config.parent_dir, f)) and f.startswith('train')]

# test_dirs = [os.path.join(Config.parent_dir, f) 
#                  for f in os.listdir(Config.parent_dir) 
#                  if os.path.isdir(os.path.join(Config.parent_dir, f)) and f.startswith('test')]

# train_parquet_files = [os.path.join(l_dir, f) 
#                  for l_dir in train_dirs for f in os.listdir(l_dir) 
#                  if f.endswith('.parquet')]

# test_parquet_files = [os.path.join(l_dir, f) 
#                  for l_dir in test_dirs for f in os.listdir(l_dir) 
#                  if f.endswith('.parquet')]

# Create dataset and dataloader
dataset = ParquetAudioDataset(train_sample_names, processor)
dataloader = DataLoader(
    dataset, 
    batch_size=Config.batch_size, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=4  # Use multiple workers for parallel loading
)


test_dataset = ParquetAudioDataset(test_sample_names, processor)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=Config.batch_size, 
    shuffle=False, 
    collate_fn=collate_fn,
    num_workers=4  # Use multiple workers for parallel loading
)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate)

# Training loop
def train():
    test()
    model.train()
    
    for epoch in range(Config.num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_features = batch['input_features'].to(Config.device)
            labels = batch['labels'].to(Config.device)
            
            # Forward pass
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Print epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if not os.path.exists(Config.save_dir):
            os.makedirs(Config.save_dir)
        torch.save(model.state_dict(), os.path.join(Config.save_dir, f"whisper_epoch_{epoch+1}.pt"))
        
        test()

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred_str, ref_str):
    wer = wer_metric.compute(predictions=pred_str, references=ref_str)
    cer = cer_metric.compute(predictions=pred_str, references=ref_str)
    return {
        "wer": wer,
        "cer": cer,
        "samples": len(pred_str)
    }   

def test():
    model.eval()
    total_loss = 0
    all_preds=[]
    all_refs=[]
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_features = batch['input_features'].to(Config.device)
            labels = batch['texts']#.to(Config.device)
            
            generated_ids = model.generate(input_features=input_features)
            
            # Decode predictions
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # outputs = model(input_features=input_features, labels=labels)
            
            
            all_preds.extend(preds)
            all_refs.extend(labels)
        all_preds = [pred.lower() for pred in all_preds]
        all_refs = [ref.lower() for ref in all_refs]
        metrics = compute_metrics(all_preds, all_refs)
            # loss = outputs.loss
            
            # total_loss += loss.item()
    print("metrics::", metrics)        
    model.train()
    return metrics
    # avg_loss = total_loss / len(test_dataloader)
    # print(f"Test Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # test()
    train()