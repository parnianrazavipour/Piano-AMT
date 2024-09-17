from pathlib import Path
from typing import Optional
import torch.nn.functional as F
import torch
import torch.nn as nn
import gc
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import librosa
import pandas as pd
import numpy as np
import random
from tqdm import tqdm  # Import tqdm for the progress bar
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import ToMono
from audidata.transforms.midi import PianoRoll
from audidata.io.midi import read_single_track_midi
from audidata.collate.base import collate_list_fn
import matplotlib.pyplot as plt
import soundfile
from torch.utils.data import DataLoader
from tqdm import tqdm 
from data_loader import MAESTRO
from crnn import CRNN




import os
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Load MAESTRO data loader and CRNN model
from data_loader import MAESTRO
from crnn import CRNN

device = "cuda"
sr = 16000


def load_maestro():
    root = "/datasets/maestro-v3.0.0"

    # Dataset
    train_dataset = MAESTRO(
        root=root,
        split="train",
        sr=sr
    )

    test_dataset = MAESTRO(
        root=root,
        split="test",
        sr=sr
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=16, 
        num_workers=16, 
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=16, 
        num_workers=16, 
    )
    
    return test_dataloader, train_dataloader



from data_loader import Slakh2100

def load_slakh2100():
    root = "/datasets/slakh2100_flac"

    # Dataset
    train_dataset = Slakh2100(
        root=root,
        split="train",
        sr=sr
    )

    test_dataset = Slakh2100(
        root=root,
        split="test",
        sr=sr
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=32, 
        num_workers=16, 
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=32, 
        num_workers=16, 
    )
    
    return test_dataloader, train_dataloader



from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_correct_onsets = 0
    total_predicted_onsets = 0
    total_actual_onsets = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            audio = data["audio"].to(device)
            onset_roll = data["frame_roll"].to(device)
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio.cpu().numpy(), sr=sr, n_fft=2048, hop_length=160, n_mels=229, fmin=0, fmax=8000
            )
            mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)
            mel_spectrogram = torch.tensor(mel_spectrogram).to(device)
            output = model(mel_spectrogram)

            loss = criterion(output, onset_roll)
            total_loss += loss.item()

            predicted_frames = (output > 0.6).float()  # may change

            # TPs
            correct_onsets = ((predicted_frames == 1) & (onset_roll == 1)).float().sum()
            predicted_onsets = predicted_frames.sum()  # All predicted onsets (1s)
            actual_onsets = onset_roll.sum()  # All actual onsets (1s)

            total_correct_onsets += correct_onsets
            total_predicted_onsets += predicted_onsets
            total_actual_onsets += actual_onsets

            all_targets.append(onset_roll.cpu().numpy())
            all_predictions.append(predicted_frames.cpu().numpy())

    # Concatenate all predictions and targets for metric calculation
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()

    # Calculate precision, recall, and F1-score
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    # Average loss
    avg_loss = total_loss / len(dataloader)
    # Onset accuracy
    onset_accuracy = total_correct_onsets / total_actual_onsets if total_actual_onsets > 0 else 0

    print(f"Test acc: {onset_accuracy:0.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    return avg_loss, onset_accuracy, precision, recall, f1



# # Updated inspect_predictions function
# def inspect_predictions(model, dataloader, num_samples=3):
#     model.eval()
#     device = next(model.parameters()).device

#     with torch.no_grad():
#         for step, data in tqdm(enumerate(dataloader), desc="Inspecting Predictions", total=num_samples):
#             if step >= num_samples:
#                 break

#             audio = data["audio"].to(device)
#             onset_roll = data["frame_roll"].to(device)

#             # Compute spectrogram
#             mel_spectrogram = librosa.feature.melspectrogram(
#                 y=audio.cpu().numpy(),
#                 sr=sr,
#                 n_fft=2048,
#                 hop_length=160,
#                 n_mels=229,
#                 fmin=0,
#                 fmax=8000
#             )
#             mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)
#             mel_spectrogram = torch.tensor(mel_spectrogram).to(device)

#             # Get predictions from model
#             output = model(mel_spectrogram)
#             predicted_onsets = (output > 0.6).cpu().numpy()

#             # Print the indices of positive numbers
#             pred_indices = np.argwhere(predicted_onsets[0] > 0)
#             gt_indices = np.argwhere(onset_roll.cpu().numpy()[0] > 0)
#             print(f"\n--- Sample {step} ---")
#             print("Predicted Onset Indices:\n", pred_indices)
#             print("Ground Truth Onset Indices:\n", gt_indices)


def visualize_predictions(model, dataloader, save_dir="visualizations", num_samples=3):
    model.eval()
    device = next(model.parameters()).device

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for step, data in tqdm(enumerate(dataloader), desc="Visualizing Predictions", total=num_samples):
            if step >= num_samples:
                break

            audio = data["audio"].to(device)
            onset_roll = data["frame_roll"].to(device)

            # Compute spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio.cpu().numpy(),
                sr=sr,
                n_fft=2048,
                hop_length=160,
                n_mels=229,
                fmin=0,
                fmax=8000
            )
            mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)
            mel_spectrogram = torch.tensor(mel_spectrogram).to(device)
            output = model(mel_spectrogram)
            predicted_onsets = (output > 0.6).cpu().numpy()

            for i in range(len(audio)):  
                fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

                mel_spectrogram_single = mel_spectrogram[i, 0, :, :] 
                axs[0].imshow(librosa.power_to_db(mel_spectrogram_single, ref=np.max), origin='lower', aspect='auto')
                axs[0].set_title(f"Mel Spectrogram (Sample {step+1}, Instance {i+1})")
                axs[0].set_ylabel("Mel Frequencies")

                time_frames = np.arange(predicted_onsets[i].shape[0])

                axs[1].stem(time_frames, np.sum(predicted_onsets[i], axis=1), linefmt='r-', markerfmt='ro', basefmt=' ')
                axs[1].set_title("Predicted Onsets")
                axs[1].set_ylabel("Onsets (Sum Over Notes)")

                axs[2].stem(time_frames, np.sum(onset_roll[i].cpu().numpy(), axis=1), linefmt='g-', markerfmt='go', basefmt=' ')
                axs[2].set_title("Ground Truth Onsets")
                axs[2].set_ylabel("Onsets (Sum Over Notes)")
                axs[2].set_xlabel("Time Frames")

                plt.tight_layout()

                fig_path = os.path.join(save_dir, f"sample_{step+1}_instance_{i+1}.png")
                plt.savefig(fig_path)
                plt.close(fig)

                pred_txt_path = os.path.join(save_dir, f"predicted_onsets_sample_{step+1}_instance_{i+1}.txt")
                gt_txt_path = os.path.join(save_dir, f"ground_truth_onsets_sample_{step+1}_instance_{i+1}.txt")

                np.savetxt(pred_txt_path, predicted_onsets[i], fmt="%d")
                np.savetxt(gt_txt_path, onset_roll[i].cpu().numpy(), fmt="%d")

                print(f"Saved visualization to {fig_path}")
                print(f"Saved predicted onsets to {pred_txt_path}")
                print(f"Saved ground truth onsets to {gt_txt_path}")



def train_maestro(epochs=10):
    test_dataloader, train_dataloader = load_maestro()
    model = CRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    total_frames = sum([data["frame_roll"].numel() for data in train_dataloader])
    onset_frames = sum([data["frame_roll"].sum().item() for data in train_dataloader])
    pos_weight = (total_frames - onset_frames) / onset_frames
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    debug_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_correct_onsets = 0
        total_predicted_onsets = 0
        total_actual_onsets = 0
        total_silent_frames = 0 
        total_frames = 0  

        for step, data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            audio = data["audio"].to(device)
            onset_roll = data["frame_roll"].to(device)

            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio.cpu().numpy(), sr=sr, n_fft=2048, hop_length=160, n_mels=229, fmin=0, fmax=8000
            )
            mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)
            mel_spectrogram = torch.tensor(mel_spectrogram).to(device)

            output = model(mel_spectrogram)

            # loss = F.binary_cross_entropy(output, onset_roll)
            loss = criterion(output, onset_roll)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            silent_frames = (onset_roll.sum(dim=-1) == 0).sum().item()  # silent frames per batch
            total_silent_frames += silent_frames
            total_frames += onset_roll.size(0) * onset_roll.size(1)  # Total number of frames in the batch

            predicted_frames = (output > 0.6).float()


            # Only count onsets (1s) in both predicted and actual
            correct_onsets = ((predicted_frames == 1) & (onset_roll == 1)).float().sum()
            predicted_onsets = predicted_frames.sum() 
            actual_onsets = onset_roll.sum()  

            total_correct_onsets += correct_onsets
            total_predicted_onsets += predicted_onsets
            total_actual_onsets += actual_onsets

        avg_train_loss = running_loss / len(train_dataloader)
        train_onset_accuracy = total_correct_onsets / total_actual_onsets if total_actual_onsets > 0 else 0

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Onset Accuracy: {train_onset_accuracy:.4f}")
        print(f"Total silent frames: {total_silent_frames} / {total_frames} ({(total_silent_frames / total_frames) * 100:.2f}% silent frames)")

        evaluate(model, test_dataloader, F.binary_cross_entropy)
        # print(f"Test Loss: {test_loss:.4f}, Test Onset Accuracy: {test_onset_accuracy:.4f}")

        visualize_predictions(model, test_dataloader, num_samples=1)

    
def train_slakh2100(epochs=3):
    test_dataloader, train_dataloader = load_slakh2100()
    model = CRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_frames = sum([data["frame_roll"].numel() for data in train_dataloader])
    onset_frames = sum([data["frame_roll"].sum().item() for data in train_dataloader])
    pos_weight = (total_frames - onset_frames) / onset_frames
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_correct_onsets = 0
        total_predicted_onsets = 0
        total_actual_onsets = 0
        total_silent_frames = 0 
        total_frames = 0 

        for step, data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            audio = data["audio"].to(device)
            frame_roll = data["frame_roll"].to(device)
    

            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio.cpu().numpy(), sr=sr, n_fft=2048, hop_length=160, n_mels=229, fmin=0, fmax=8000
            )
            mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)
            mel_spectrogram = torch.tensor(mel_spectrogram).to(device)

            output = model(mel_spectrogram)

            loss = criterion(output, frame_roll)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            silent_frames = (frame_roll.sum(dim=-1) == 0).sum().item()
            total_silent_frames += silent_frames
            total_frames += frame_roll.size(0) * frame_roll.size(1)

            predicted_frames = (output > 0.5).float()
            correct_onsets = ((predicted_frames == 1) & (frame_roll == 1)).float().sum()
            predicted_onsets = predicted_frames.sum()
            actual_onsets = frame_roll.sum()

            total_correct_onsets += correct_onsets
            total_predicted_onsets += predicted_onsets
            total_actual_onsets += actual_onsets

        avg_train_loss = running_loss / len(train_dataloader)
        train_onset_accuracy = total_correct_onsets / total_actual_onsets if total_actual_onsets > 0 else 0

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Onset Accuracy: {train_onset_accuracy:.4f}")
        print(f"Total silent frames: {total_silent_frames} / {total_frames} ({(total_silent_frames / total_frames) * 100:.2f}% silent frames)")

        # Evaluate on the test set
        test_loss, test_onset_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_dataloader, criterion)
        print(f"Test Loss: {test_loss:.4f}, Test Onset Accuracy: {test_onset_accuracy:.4f}")

if __name__ == "__main__":
    torch.cuda.empty_cache()

    gc.collect()

    train_maestro()

    
