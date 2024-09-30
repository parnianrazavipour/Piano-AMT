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
import torchaudio.transforms as T

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

# # Load MAESTRO data loader and CRNN model
# from data_loader import MAESTRO
# from crnn import CRNN

# device = "cuda"
# sr = 16000


# def load_maestro():
#     root = "/datasets/maestro-v3.0.0"

#     # Dataset
#     train_dataset = MAESTRO(
#         root=root,
#         split="train",
#         sr=sr
#     )

#     test_dataset = MAESTRO(
#         root=root,
#         split="test",
#         sr=sr
#     )

#     train_dataloader = DataLoader(
#         dataset=train_dataset, 
#         batch_size=16, 
#         num_workers=16, 
#     )

#     test_dataloader = DataLoader(
#         dataset=test_dataset, 
#         batch_size=16, 
#         num_workers=16, 
#     )
    
#     return test_dataloader, train_dataloader



# from data_loader import Slakh2100

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



# from sklearn.metrics import precision_score, recall_score, f1_score

# def evaluate(model, dataloader, criterion):
#     model.eval()
#     total_loss = 0
#     total_correct_onsets = 0
#     total_predicted_onsets = 0
#     total_actual_onsets = 0
#     all_targets = []
#     all_predictions = []

#     with torch.no_grad():
#         for data in tqdm(dataloader, desc="Evaluating"):
#             audio = data["audio"].to(device)
#             onset_roll = data["onset_roll"].to(device)
#             mel_spectrogram_transform = T.MelSpectrogram(
#                 sample_rate=sr,
#                 n_fft=2048,
#                 hop_length=160,
#                 n_mels=229,
#                 f_min=0,
#                 f_max=8000,
#                 power=2.0
#             ).to(device) 

#             # Compute the Mel spectrogram using torchaudio
#             mel_spectrogram = mel_spectrogram_transform(audio)

#             # Apply logarithmic scaling
#             mel_spectrogram = torch.log1p(mel_spectrogram)

#             # mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()



#             # # Normalize the spectrogram
#             # mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()

#             output = model(mel_spectrogram)

#             loss = criterion(output, onset_roll)
#             total_loss += loss.item()

#             # output_probs = torch.sigmoid(output)

#             # For thresholding
#             predicted_frames = (output > 0.65).float()

#             # TPs
#             correct_onsets = ((predicted_frames == 1) & (onset_roll == 1)).float().sum()
#             predicted_onsets = predicted_frames.sum()  # All predicted onsets (1s)
#             actual_onsets = onset_roll.sum()  # All actual onsets (1s)

#             total_correct_onsets += correct_onsets
#             total_predicted_onsets += predicted_onsets
#             total_actual_onsets += actual_onsets

#             all_targets.append(onset_roll.cpu().numpy())
#             all_predictions.append(predicted_frames.cpu().numpy())

#     # Concatenate all predictions and targets for metric calculation
#     all_targets = np.concatenate(all_targets, axis=0).flatten()
#     all_predictions = np.concatenate(all_predictions, axis=0).flatten()

#     # Calculate precision, recall, and F1-score
#     precision = precision_score(all_targets, all_predictions)
#     recall = recall_score(all_targets, all_predictions)
#     f1 = f1_score(all_targets, all_predictions)

#     # Average loss
#     avg_loss = total_loss / len(dataloader)
#     # Onset accuracy
#     onset_accuracy = total_correct_onsets / total_actual_onsets if total_actual_onsets > 0 else 0

#     print(f"Test acc: {onset_accuracy:0.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

#     return avg_loss, onset_accuracy, precision, recall, f1

#     # Find the optimal threshold for F1 score
#     # best_f1 = 0
#     # best_threshold = 0.65
#     # for threshold in np.arange(0.65, 0.65, 0.65):
#     #     preds = (all_predictions > threshold).astype(int)
#     #     f1 = f1_score(all_targets, preds)
#     #     if f1 > best_f1:
#     #         best_f1 = f1
#     #         best_threshold = threshold

#     # # Calculate metrics with the best threshold
#     # predicted_frames = (all_predictions > best_threshold).astype(int)
#     # precision = precision_score(all_targets, predicted_frames)
#     # recall = recall_score(all_targets, predicted_frames)
#     # f1 = f1_score(all_targets, predicted_frames)

#     # # Average loss
#     # avg_loss = np.mean(losses)

#     # # Onset accuracy
#     # onset_accuracy = recall  # Since recall = TP / (TP + FN)

#     # print(f"Validation Loss: {avg_loss:.4f}, F1-Score: {f1:.4f} (Threshold: {best_threshold})")

#     # return avg_loss, onset_accuracy, precision, recall, f1

# # # Updated inspect_predictions function
# # def inspect_predictions(model, dataloader, num_samples=3):
# #     model.eval()
# #     device = next(model.parameters()).device

# #     with torch.no_grad():
# #         for step, data in tqdm(enumerate(dataloader), desc="Inspecting Predictions", total=num_samples):
# #             if step >= num_samples:
# #                 break

# #             audio = data["audio"].to(device)
# #             onset_roll = data["onset_roll"].to(device)

# #             # Compute spectrogram
# #             mel_spectrogram = librosa.feature.melspectrogram(
# #                 y=audio.cpu().numpy(),
# #                 sr=sr,
# #                 n_fft=2048,
# #                 hop_length=160,
# #                 n_mels=229,
# #                 fmin=0,
# #                 fmax=8000
# #             )
# #             mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)
# #             mel_spectrogram = torch.tensor(mel_spectrogram).to(device)

# #             # Get predictions from model
# #             output = model(mel_spectrogram)
# #             predicted_onsets = (output > 0.65).cpu().numpy()

# #             # Print the indices of positive numbers
# #             pred_indices = np.argwhere(predicted_onsets[0] > 0)
# #             gt_indices = np.argwhere(onset_roll.cpu().numpy()[0] > 0)
# #             print(f"\n--- Sample {step} ---")
# #             print("Predicted Onset Indices:\n", pred_indices)
# #             print("Ground Truth Onset Indices:\n", gt_indices)

def visualize_predictions(model, dataloader, save_dir="visualizations", num_samples=3):
    model.eval()
    device = next(model.parameters()).device

    # Create directory to save the visualizations and vectors
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for step, data in tqdm(enumerate(dataloader), desc="Visualizing Predictions", total=num_samples):
            if step >= num_samples:
                break

            audio = data["audio"].to(device)
            onset_roll = data["onset_roll"].to(device)

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
            mel_spectrogram_ = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)



            mel_spectrogram_transform = T.MelSpectrogram(
                sample_rate=sr,
                n_fft=2048,
                hop_length=160,
                n_mels=229,
                f_min=0,
                f_max=8000,
                power=2.0
            ).to(device) 

            # Compute the Mel spectrogram using torchaudio
            mel_spectrogram = mel_spectrogram_transform(audio)

            # Apply logarithmic scaling
            mel_spectrogram = torch.log1p(mel_spectrogram)

            mel_spectrogram = add_gaussian_noise(mel_spectrogram)

            # Normalize the spectrogram
            mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()

            output = model(mel_spectrogram)
            predicted_onsets = (output > 0.65).cpu().numpy()

            # Visualize the spectrogram, predicted onsets, and ground truth
            for i in range(len(audio)):  # Loop over each sample in the batch
                fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

                # Plot the Mel Spectrogram (using a color map for better clarity)
                mel_spectrogram_single = mel_spectrogram_[i, 0, :, :]  # Remove extra dimensions
                axs[0].imshow(librosa.power_to_db(mel_spectrogram_single, ref=np.max), origin='lower', aspect='auto')
                axs[0].set_title(f"Mel Spectrogram (Sample {step+1}, Instance {i+1})")
                axs[0].set_ylabel("Mel Frequencies")

                # Convert predicted_onsets and ground truth onsets to stem plots
                time_frames = np.arange(predicted_onsets[i].shape[0])

                # Plot predicted onsets using stem plot
                axs[1].stem(time_frames, np.sum(predicted_onsets[i], axis=1), linefmt='r-', markerfmt='ro', basefmt=' ')
                axs[1].set_title("Predicted Onsets")
                axs[1].set_ylabel("Onsets (Sum Over Notes)")

                # Plot ground truth onsets using stem plot
                axs[2].stem(time_frames, np.sum(onset_roll[i].cpu().numpy(), axis=1), linefmt='g-', markerfmt='go', basefmt=' ')
                axs[2].set_title("Ground Truth Onsets")
                axs[2].set_ylabel("Onsets (Sum Over Notes)")
                axs[2].set_xlabel("Time Frames")

                plt.tight_layout()

                # Save the figure
                fig_path = os.path.join(save_dir, f"sample_{step+1}_instance_{i+1}.png")
                plt.savefig(fig_path)
                plt.close(fig)

                # Save the predicted onsets and ground truth as text files
                pred_txt_path = os.path.join(save_dir, f"predicted_onsets_sample_{step+1}_instance_{i+1}.txt")
                gt_txt_path = os.path.join(save_dir, f"ground_truth_onsets_sample_{step+1}_instance_{i+1}.txt")

                np.savetxt(pred_txt_path, predicted_onsets[i], fmt="%d")
                np.savetxt(gt_txt_path, onset_roll[i].cpu().numpy(), fmt="%d")

                print(f"Saved visualization to {fig_path}")
                print(f"Saved predicted onsets to {pred_txt_path}")
                print(f"Saved ground truth onsets to {gt_txt_path}")


# # class FocalLoss(nn.Module):
# #     def __init__(self, gamma=2, alpha=None):
# #         super(FocalLoss, self).__init__()
# #         self.gamma = gamma
# #         self.alpha = alpha

# #     def forward(self, inputs, targets):
# #         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
# #         pt = torch.exp(-BCE_loss)
# #         F_loss = ((1 - pt) ** self.gamma) * BCE_loss
# #         if self.alpha:
# #             alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
# #             F_loss = alpha_t * F_loss
# #         return F_loss.mean()
                




# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=None):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha

#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = ((1 - pt) ** self.gamma) * BCE_loss
#         if self.alpha:
#             alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
#             F_loss = alpha_t * F_loss
#         return F_loss.mean()



# import torch
# import os
# import torch.optim.lr_scheduler as lr_scheduler

# def train_maestro(epochs=30, save_path="model_checkpoint.pth"):
#     test_dataloader, train_dataloader = load_maestro()
#     model = CRNN().to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

#     scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR by a factor of 0.1 every 10 epochs

#     total_frames = sum([data["onset_roll"].numel() for data in train_dataloader])
#     onset_frames = sum([data["onset_roll"].sum().item() for data in train_dataloader])
#     pos_weight = (total_frames - onset_frames) / onset_frames
#     # criterion = nn.BCELoss()  # Use BCELoss when outputting probabilities

    


#     # criterion = FocalLoss(gamma=2.0, alpha=0.655).to(device)


#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         total_correct_onsets = 0
#         total_predicted_onsets = 0
#         total_actual_onsets = 0
#         total_silent_frames = 0 
#         total_frames = 0  

#         for step, data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
#             audio = data["audio"].to(device)
#             onset_roll = data["onset_roll"].to(device)

#             mel_spectrogram_transform = T.MelSpectrogram(
#                 sample_rate=sr,
#                 n_fft=2048,
#                 hop_length=160,
#                 n_mels=229,
#                 f_min=0,
#                 f_max=8000,
#                 power=2.0
#             ).to(device) 

#             # Compute the Mel spectrogram using torchaudio
#             mel_spectrogram = mel_spectrogram_transform(audio)

#             # Apply logarithmic scaling
#             mel_spectrogram = torch.log1p(mel_spectrogram)


#             # mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()



#             # # Normalize the spectrogram
#             # mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()

#             # mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)
#             # mel_spectrogram = torch.tensor(mel_spectrogram).to(device)
#             # mel_spectrogram = torch.log10(torch.clamp(mel_spectrogram, min=1e-10))

#             output = model(mel_spectrogram)

#             loss = criterion(output, onset_roll)
#             running_loss += loss.item()

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             silent_frames = (onset_roll.sum(dim=-1) == 0).sum().item()  # silent frames per batch
#             total_silent_frames += silent_frames
#             total_frames += onset_roll.size(0) * onset_roll.size(1)  # Total number of frames in the batch

#             predicted_frames = (output > 0.65).float()


#             correct_onsets = ((predicted_frames == 1) & (onset_roll == 1)).float().sum()
#             predicted_onsets = predicted_frames.sum() 
#             actual_onsets = onset_roll.sum()  

#             total_correct_onsets += correct_onsets
#             total_predicted_onsets += predicted_onsets
#             total_actual_onsets += actual_onsets

#         avg_train_loss = running_loss / len(train_dataloader)
#         train_onset_accuracy = total_correct_onsets / total_actual_onsets if total_actual_onsets > 0 else 0

#         print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Onset Accuracy: {train_onset_accuracy:.4f}")
#         print(f"Total silent frames: {total_silent_frames} / {total_frames} ({(total_silent_frames / total_frames) * 100:.2f}% silent frames)")

#         evaluate(model, test_dataloader, nn.BCEWithLogitsLoss())
#         visualize_predictions(model, test_dataloader, num_samples=1)

#         # Step the scheduler to adjust the learning rate after every epoch
#         scheduler.step()

#     # After all epochs, save the model and optimizer state_dict
#     print("Saving model...")
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'epoch': epochs
#     }, save_path)
#     print(f"Model saved to {save_path}")


    
def train_slakh2100(epochs=3):
    test_dataloader, train_dataloader = load_slakh2100()
    model = CRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_frames = sum([data["onset_roll"].numel() for data in train_dataloader])
    onset_frames = sum([data["onset_roll"].sum().item() for data in train_dataloader])
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
            frame_roll = data["onset_roll"].to(device)
    

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

            predicted_frames = (output > 0.65).float()
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

# if __name__ == "__main__":
#     torch.cuda.empty_cache()

#     gc.collect()

#     train_maestro()

    




# import torch
# import os
# import gc
# import torch.optim as optim
# import torch.nn.functional as F
# from tqdm import tqdm
# import torchaudio.transforms as T
# from sklearn.metrics import precision_score, recall_score, f1_score
# from data_loader import MAESTRO, CRNN

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
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=16, 
        num_workers=16, 
        pin_memory=True
    )
    
    return test_dataloader, train_dataloader


def add_gaussian_noise(input_tensor, mean=0.0, std=0.05):
    noise = torch.randn(input_tensor.size()).to(input_tensor.device) * std + mean
    return input_tensor + noise




# def add_gaussian_noise(input_tensor, mean= 0.0, std=0.05):
#         noise = torch.randn(input_tensor.size()).to(input_tensor.device) * std + mean
#         return input_tensor + noise
    


class InfiniteSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.pointer = 0
        random.shuffle(self.indices)

    def __iter__(self):
        while True:
            if self.pointer >= len(self.indices):
                random.shuffle(self.indices)
                self.pointer = 0
            yield self.indices[self.pointer]
            self.pointer += 1


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, use_logits=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_logits = use_logits

    def forward(self, logits, targets):
        if self.use_logits:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

    
def train_maestro(steps=10000, save_path="model_checkpoint.pth", test_step_frequency=100, save_step_frequency=100 , onset_tolerance = 0.01 , fps = 100):
    test_dataloader, train_dataloader = load_maestro()
    model = CRNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    total_frames = sum([data["onset_roll"].numel() for data in train_dataloader])
    onset_frames = sum([data["onset_roll"].sum().item() for data in train_dataloader])
    pos_weight = (total_frames - onset_frames) / (onset_frames)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    # criterion = BinaryFocalLoss().to(device)
    # criterion = nn.BCELoss().to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

    train_sampler = InfiniteSampler(train_dataloader.dataset)
    train_loader = DataLoader(train_dataloader.dataset, batch_size=8, sampler=train_sampler, num_workers=16)
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=160,
        n_mels=229,
        f_min=0,
        f_max=8000,
        power=2.0,
        normalized=True,
    ).to(device)

    step = 0
    model.train()  # Ensure model is in training mode
    while step < steps:
        running_loss = 0.0
        total_correct_onsets = 0
        total_predicted_onsets = 0
        total_actual_onsets = 0
        total_silent_frames = 0
        total_frames = 0
        all_targets = []
        all_predictions = []

        for data in tqdm(train_loader, desc=f"Step {step}/{steps}"):
            if step >= steps:
                break

            audio = data["audio"].to(device)
            onset_roll = data["onset_roll"].to(device)
            frame_roll = onset_roll

            mel_spectrogram = mel_spectrogram_transform(audio)
            mel_spectrogram = torch.log1p(mel_spectrogram)
            mel_spectrogram = add_gaussian_noise(mel_spectrogram)
            mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()

            output = model(mel_spectrogram)
            loss = criterion(output, onset_roll)
            running_loss += loss.item()

            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()       

            predicted_frames = (output > 0.65).float().cpu().numpy()
            true_frames = frame_roll.cpu().numpy()

            # Match predicted onsets with true onsets within the tolerance window
            predicted_frames_flat = predicted_frames.flatten()
            true_frames_flat = true_frames.flatten()
            
            matched_pred_onsets, matched_true_onsets = match_onsets_with_tolerance(
                predicted_frames_flat, true_frames_flat, tolerance=onset_tolerance, fps=fps
            )

            all_targets.append(matched_true_onsets)
            all_predictions.append(matched_pred_onsets)

            # Silent frames count
            silent_frames = (frame_roll.sum(dim=-1) == 0).sum().item()
            total_silent_frames += silent_frames
            total_frames += frame_roll.size(0) * frame_roll.size(1)

            step += 1

            if step % test_step_frequency == 0:
                all_targets_flat = np.concatenate(all_targets, axis=0).flatten()
                all_predictions_flat = np.concatenate(all_predictions, axis=0).flatten()

                train_accuracy = np.sum(all_targets_flat == all_predictions_flat) / len(all_targets_flat)

                avg_train_loss = running_loss / step
                print(f"Step [{step}/{steps}] - Train Loss: {avg_train_loss:.4f}, Train Onset Accuracy: {train_accuracy:.4f}")
                print(f"Total silent frames: {total_silent_frames} / {total_frames} ({(total_silent_frames / total_frames) * 100:.2f}% silent frames)")

                model.eval()  # Set model to evaluation mode
                test_loss, onset_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_dataloader, criterion, fps=fps, onset_tolerance=onset_tolerance)
                model.train()  # Switch back to training mode after evaluation

        scheduler.step()

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'steps': step
    }, save_path)




import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

def match_onsets_with_tolerance(pred_onsets, true_onsets, tolerance, fps):
    matched_pred_onsets = np.zeros_like(pred_onsets)
    matched_true_onsets = np.zeros_like(true_onsets)

    # Convert tolerance to frames
    tolerance_frames = int(tolerance * fps)

    # Loop over true onsets and find matching predicted onsets within tolerance
    for i, true_onset in enumerate(true_onsets):
        if true_onset == 1:
            for j in range(max(0, i - tolerance_frames), min(len(pred_onsets), i + tolerance_frames + 1)):
                if pred_onsets[j] == 1:
                    matched_pred_onsets[j] = 1
                    matched_true_onsets[i] = 1
                    break

    return matched_pred_onsets, matched_true_onsets

def evaluate(model, dataloader, criterion, fps=100, onset_tolerance=0.01):
    model.eval()  # Set to evaluation mode
    total_loss = 0
    total_correct_onsets = 0
    total_predicted_onsets = 0
    total_actual_onsets = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            audio = data["audio"].to(device)
            onset_roll = data["onset_roll"].to(device)
            mel_spectrogram_transform = T.MelSpectrogram(
                sample_rate=sr,
                n_fft=2048,
                hop_length=160,
                n_mels=229,
                f_min=0,
                f_max=8000,
                power=2.0,
                normalized=True
            ).to(device)

            # Compute the Mel spectrogram using torchaudio
            mel_spectrogram = mel_spectrogram_transform(audio)

            # Apply logarithmic scaling
            mel_spectrogram = torch.log1p(mel_spectrogram)
            mel_spectrogram = add_gaussian_noise(mel_spectrogram)
            mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()

            output = model(mel_spectrogram)

            loss = criterion(output, onset_roll)
            total_loss += loss.item()

            predicted_frames = (output > 0.65).float().cpu().numpy()
            true_frames = onset_roll.cpu().numpy()

            # Flatten arrays for precision/recall calculation
            predicted_frames_flat = predicted_frames.flatten()
            true_frames_flat = true_frames.flatten()

            # Match predicted onsets with true onsets within the tolerance window
            matched_pred_onsets, matched_true_onsets = match_onsets_with_tolerance(
                predicted_frames_flat, true_frames_flat, tolerance=onset_tolerance, fps=fps
            )

            # Append matched predictions and targets for metric calculation
            all_targets.append(matched_true_onsets)
            all_predictions.append(matched_pred_onsets)

    # Concatenate all predictions and targets for metric calculation
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()

    # Calculate precision, recall, and F1-score with the matched onsets
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    # Average loss
    avg_loss = total_loss / len(dataloader)
    # Onset accuracy
    onset_accuracy = np.sum(all_targets == all_predictions) / len(all_targets)

    print(f"Test Loss: {avg_loss:.4f} Test acc: {onset_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    return avg_loss, onset_accuracy, precision, recall, f1

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    train_maestro()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    train_maestro()
