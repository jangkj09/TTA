
import os
import random
from datasets import Dataset, DatasetDict, Audio, load_from_disk
from sklearn.model_selection import train_test_split
from transformers import WhisperProcessor
import torch
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def random_augment(audio, sample_rate):
    x = audio.to(torch.float32)

    if random.random() < 0.5:
        # Add gain
        gain = random.uniform(0.8, 1.2)
        x *= gain

    if random.random() < 0.5:
        # Resample to simulate speed change
        new_sr = int(sample_rate * random.uniform(0.9, 1.1))
        resample = T.Resample(orig_freq=sample_rate, new_freq=new_sr)
        x = resample(x)

    if random.random() < 0.5:
        lowpass = T.Vol(gain=random.uniform(0.5, 1.0))
        x = lowpass(x)

    return x

def prepare_dataset_train(example, idx):
    try:

        audio = example["audio"]
        text = example["transcription"]
        waveform = torch.tensor(audio["array"])
        sample_rate = audio["sampling_rate"]

        # Create two differently augmented views
        aug1 = random_augment(waveform, sample_rate).squeeze(0).numpy()
        aug2 = random_augment(waveform, sample_rate).squeeze(0).numpy()

        # Process with processor
        processed_1 = processor(audio=aug1, sampling_rate=sample_rate, text=text)
        processed_2 = processor(audio=aug2, sampling_rate=sample_rate, text=text)
        example = {
            "input_features": np.concatenate([processed_1["input_features"],processed_2["input_features"]], axis=1),
            "labels": processed_2["labels"]
        }
        example["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return example
    except Exception as e:
        print(f"Error processing example {idx}: {e}")
        print("Exception:", e)
        return None
    
def prepare_dataset_test(example, idx):
    try:
        audio = example["audio"]
        text = example["transcription"]
        example = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=text,
        )
        example["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return example
    except Exception as e:
        print(f"Error processing example {idx}: {e}")
        print("Exception:", e)
        return None

def is_audio_in_length_range(length):
    return length < max_input_length

if __name__ == "__main__":
    # Define the path to the dataset
    dataset_path = "./"  # Update this path
    dataset_all= load_from_disk("openSLR-ES")

    processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="spanish", task="transcribe"
    )

    common_voice = DatasetDict()
    common_voice["train"] = dataset_all["train"]
    common_voice["test"] = dataset_all["test"]

    num_threads = min(32, (os.cpu_count() or 1) + 4)
    print(num_threads)

    common_voice["train"] = common_voice["train"].map(prepare_dataset_train,
                                                        with_indices=True,
                                                        num_proc=num_threads
                                                        )
    max_input_length = 30.0

    common_voice["train"] = common_voice["train"].filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )
    



    common_voice["test"] = common_voice["test"].map(prepare_dataset_test,
                                                        with_indices=True,
                                                        num_proc=num_threads
                                                        )
    common_voice.save_to_disk("openSLR-ES-processed-bt")
    print('Done processing openSLR-ES for bt training.')


