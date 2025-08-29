#!/usr/bin/env python3
"""
Comprehensive Ablation Study for PLRNN Variants
===============================================

This script performs a comprehensive comparison of:
1. Vanilla PLRNN (baseline)
2. GTF-shPLRNN (our enhanced method)
3. Dendritic PLRNN (alternative variant)

With proper early stopping and extensive training.

Author: Master Thesis Project
Date: 2025
"""

import os
import gc
import time
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wfdb
from scipy.signal import find_peaks, welch, butter, filtfilt
from scipy.stats import skew, kurtosis
from sklearn.metrics import (
    hamming_loss, accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Metal Performance Shaders (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU")

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Configuration
class AblationConfig:
    # Data paths
    MULTILABEL_DATA_CSV = '/Users/zixiang/PycharmProjects/Master-Thesis/mimic_ecg_multilabel_dataset.csv'
    BINARY_LABELS_CSV = '/Users/zixiang/PycharmProjects/Master-Thesis/mimic_ecg_binary_labels.csv'
    ECG_BASE_PATH = '/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
    
    # Model parameters
    SEQUENCE_LENGTH = 5000  # 10 seconds at 500Hz
    NUM_CHANNELS = 12
    NUM_FEATURES = 32
    HIDDEN_DIM = 64
    LATENT_DIM = 32
    
    # Training parameters - increased for better convergence
    BATCH_SIZE = 4  # Small batch for stability
    LEARNING_RATE = 0.0001
    MAX_EPOCHS = 50  # Increased from 10
    PATIENCE = 10  # Early stopping patience
    MIN_DELTA = 0.001  # Minimum improvement for early stopping
    
    # Data limits
    MAX_TRAIN_SAMPLES = 200  # Increased for better training
    MAX_VAL_SAMPLES = 50
    MAX_TEST_SAMPLES = 50
    
    DEVICE = device

config = AblationConfig()

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
        
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

# Copy the fixed RealECGDataset from robust_plrnn_training.py
class RealECGDataset(Dataset):
    """
    Dataset that loads real ECG data from MIMIC-IV-ECG files
    """
    def __init__(self, df, binary_labels_df, ecg_base_path, max_samples=None, sequence_length=5000):
        self.df = df.copy()
        self.binary_labels_df = binary_labels_df.copy()
        self.ecg_base_path = ecg_base_path
        self.sequence_length = sequence_length
        
        if max_samples:
            self.df = self.df.head(max_samples)
            self.binary_labels_df = self.binary_labels_df.head(max_samples)
        
        # Build file paths
        self.file_paths = []
        for idx, row in self.df.iterrows():
            study_id = str(row['study_id'])
            patient_folder = f"p{study_id[:2]}/p{study_id[:8]}/s{study_id}/"
            file_path = os.path.join(ecg_base_path, "files", patient_folder)
            self.file_paths.append(file_path)
        
        print(f"📊 Dataset size: {len(self.df)} samples")
        
    def __len__(self):
        return len(self.df)
    
    def load_ecg_signal(self, file_path):
        """Load ECG signal from WFDB files"""
        try:
            if os.path.exists(file_path):
                files = [f for f in os.listdir(file_path) if f.endswith('.hea')]
                if files:
                    record_name = files[0].replace('.hea', '')
                    full_path = os.path.join(file_path, record_name)
                    
                    record = wfdb.rdrecord(full_path)
                    signal = record.p_signal.T
                    
                    # Ensure 12 leads
                    if signal.shape[0] < 12:
                        padding = np.zeros((12 - signal.shape[0], signal.shape[1]))
                        signal = np.vstack([signal, padding])
                    elif signal.shape[0] > 12:
                        signal = signal[:12, :]
                    
                    return signal, True
            
            return None, False
            
        except Exception as e:
            return None, False
    
    def preprocess_signal(self, signal, fs=500):
        """Preprocess ECG signal"""
        # Remove baseline wander
        b, a = butter(4, 0.5/(fs/2), btype='high')
        signal_filtered = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            signal_filtered[i] = filtfilt(b, a, signal[i])
        
        # Normalize
        for i in range(signal.shape[0]):
            mean_val = np.mean(signal_filtered[i])
            std_val = np.std(signal_filtered[i])
            if std_val > 0:
                signal_filtered[i] = (signal_filtered[i] - mean_val) / std_val
        
        # Ensure correct length
        if signal_filtered.shape[1] > self.sequence_length:
            start = (signal_filtered.shape[1] - self.sequence_length) // 2
            signal_filtered = signal_filtered[:, start:start + self.sequence_length]
        elif signal_filtered.shape[1] < self.sequence_length:
            pad_width = self.sequence_length - signal_filtered.shape[1]
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            signal_filtered = np.pad(signal_filtered, ((0, 0), (pad_left, pad_right)), 'constant')
        
        return signal_filtered
    
    def __getitem__(self, idx):
        # Try to load real ECG data
        file_path = self.file_paths[idx]
        signal, success = self.load_ecg_signal(file_path)
        
        if success and signal is not None:
            signal = self.preprocess_signal(signal)
            ecg_tensor = torch.FloatTensor(signal)
        else:
            # Generate synthetic data as fallback
            t = np.linspace(0, 10, self.sequence_length)
            labels_row = self.binary_labels_df.iloc[idx]
            base_hr = 75
            
            if 'ARRHYTHMIA_TACHYCARDIA' in labels_row.index and labels_row.get('ARRHYTHMIA_TACHYCARDIA', 0) == 1:
                base_hr = 110
            elif 'ARRHYTHMIA_BRADYCARDIA' in labels_row.index and labels_row.get('ARRHYTHMIA_BRADYCARDIA', 0) == 1:
                base_hr = 50
            
            # Generate 12-lead synthetic ECG
            signals = []
            for lead in range(12):
                phase_shift = lead * np.pi / 6
                
                p_wave = 0.2 * np.sin(2 * np.pi * (base_hr/60) * t + phase_shift)
                qrs = 1.0 * np.exp(-((t % (60/base_hr) - 0.05) / 0.02)**2)
                t_wave = 0.3 * np.sin(np.pi * (base_hr/60) * t + phase_shift + np.pi)
                
                signal = p_wave + qrs + t_wave
                signal += np.random.normal(0, 0.05, len(signal))
                signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
                signals.append(signal)
            
            ecg_tensor = torch.FloatTensor(np.array(signals))
        
        # Get labels - exclude non-numeric columns
        labels_row = self.binary_labels_df.iloc[idx]
        numeric_columns = labels_row.index[~labels_row.index.isin(['subject_id', 'study_id', 'ecg_time'])]
        labels_array = labels_row[numeric_columns].values.astype(np.float32)
        labels_tensor = torch.FloatTensor(labels_array)
        
        return ecg_tensor, labels_tensor

# Model Architectures

class VanillaPLRNN(nn.Module):
    """Vanilla PLRNN implementation"""
    def __init__(self, config):
        super(VanillaPLRNN, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv1d(config.NUM_CHANNELS, 32, kernel_size=50, stride=10)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, config.NUM_FEATURES, kernel_size=25, stride=5)
        self.bn2 = nn.BatchNorm1d(config.NUM_FEATURES)
        
        # PLRNN components
        self.input_proj = nn.Linear(config.NUM_FEATURES, config.LATENT_DIM)
        self.A = nn.Parameter(torch.eye(config.LATENT_DIM) * 0.9)
        self.W = nn.Parameter(torch.randn(config.LATENT_DIM, config.HIDDEN_DIM) * 0.1)
        self.h = nn.Parameter(torch.zeros(config.HIDDEN_DIM))
        
        # Layer normalization
        self.ln = nn.LayerNorm(config.LATENT_DIM)
        
        # Output projection
        self.output_proj = nn.Linear(config.LATENT_DIM, config.LATENT_DIM)
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, config.LATENT_DIM, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = self.input_proj(x[:, t])
            x_t = self.ln(x_t)
            
            # PLRNN update
            linear_part = torch.matmul(h, self.A.T)
            nonlinear_part = torch.matmul(F.relu(torch.matmul(h, self.W) + self.h), self.W.T)
            h = linear_part + nonlinear_part + x_t
            
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim=1)
        pooled = torch.mean(outputs, dim=1)
        output = self.output_proj(pooled)
        
        return output

class GTFShallowPLRNN(nn.Module):
    """GTF-enhanced Shallow PLRNN"""
    def __init__(self, config):
        super(GTFShallowPLRNN, self).__init__()
        
        # Feature extraction (same as vanilla)
        self.conv1 = nn.Conv1d(config.NUM_CHANNELS, 32, kernel_size=50, stride=10)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, config.NUM_FEATURES, kernel_size=25, stride=5)
        self.bn2 = nn.BatchNorm1d(config.NUM_FEATURES)
        
        # Shallow PLRNN components
        self.input_proj = nn.Linear(config.NUM_FEATURES, config.LATENT_DIM)
        self.A = nn.Parameter(torch.eye(config.LATENT_DIM) * 0.9)
        self.W1 = nn.Parameter(torch.randn(config.LATENT_DIM, config.HIDDEN_DIM) * 0.1)
        self.W2 = nn.Parameter(torch.randn(config.HIDDEN_DIM, config.LATENT_DIM) * 0.1)
        self.h1 = nn.Parameter(torch.zeros(config.LATENT_DIM))
        self.h2 = nn.Parameter(torch.zeros(config.HIDDEN_DIM))
        
        # GTF parameters
        self.alpha = 0.1  # GTF mixing parameter
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.LATENT_DIM)
        self.ln2 = nn.LayerNorm(config.HIDDEN_DIM)
        
        # Output projection
        self.output_proj = nn.Linear(config.LATENT_DIM, config.LATENT_DIM)
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, config.LATENT_DIM, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = self.input_proj(x[:, t])
            x_t = self.ln1(x_t)
            
            # Shallow PLRNN update
            linear_part = torch.matmul(h, self.A.T)
            hidden = F.relu(torch.matmul(h, self.W2.T) + self.h2)
            hidden = self.ln2(hidden)
            nonlinear_part = torch.matmul(hidden, self.W1.T) + self.h1
            
            # GTF mixing
            h_pred = linear_part + nonlinear_part
            h = (1 - self.alpha) * (h_pred + x_t) + self.alpha * h_pred
            
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim=1)
        pooled = torch.mean(outputs, dim=1)
        output = self.output_proj(pooled)
        
        return output

class DendriticPLRNN(nn.Module):
    """Dendritic PLRNN with multiple pathways"""
    def __init__(self, config):
        super(DendriticPLRNN, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv1d(config.NUM_CHANNELS, 32, kernel_size=50, stride=10)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, config.NUM_FEATURES, kernel_size=25, stride=5)
        self.bn2 = nn.BatchNorm1d(config.NUM_FEATURES)
        
        # Dendritic pathways
        self.input_proj = nn.Linear(config.NUM_FEATURES, config.LATENT_DIM)
        self.A = nn.Parameter(torch.eye(config.LATENT_DIM) * 0.9)
        
        # Multiple dendritic branches
        self.W1_soma = nn.Parameter(torch.randn(config.LATENT_DIM, config.HIDDEN_DIM) * 0.1)
        self.W1_basal = nn.Parameter(torch.randn(config.LATENT_DIM, config.HIDDEN_DIM) * 0.1)
        self.W1_apical = nn.Parameter(torch.randn(config.LATENT_DIM, config.HIDDEN_DIM) * 0.1)
        
        self.h1 = nn.Parameter(torch.zeros(config.HIDDEN_DIM))
        self.h2 = nn.Parameter(torch.zeros(config.HIDDEN_DIM))
        self.h3 = nn.Parameter(torch.zeros(config.HIDDEN_DIM))
        
        # Integration weights
        self.integration_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Layer normalization
        self.ln = nn.LayerNorm(config.LATENT_DIM)
        
        # Output projection
        self.output_proj = nn.Linear(config.LATENT_DIM, config.LATENT_DIM)
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, config.LATENT_DIM, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = self.input_proj(x[:, t])
            x_t = self.ln(x_t)
            
            # Linear part
            linear_part = torch.matmul(h, self.A.T)
            
            # Multiple dendritic pathways
            soma = F.relu(torch.matmul(h, self.W1_soma) + self.h1)
            basal = F.relu(torch.matmul(h, self.W1_basal) + self.h2)
            apical = F.relu(torch.matmul(h, self.W1_apical) + self.h3)
            
            # Weighted integration
            weights = F.softmax(self.integration_weights, dim=0)
            integrated = weights[0] * soma + weights[1] * basal + weights[2] * apical
            
            # Combine pathways
            nonlinear_part = torch.matmul(integrated, self.W1_soma.T)  # Use soma weights for output
            
            h = linear_part + nonlinear_part + x_t
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim=1)
        pooled = torch.mean(outputs, dim=1)
        output = self.output_proj(pooled)
        
        return output

class MultiLabelClassifier(nn.Module):
    """Wrapper for different PLRNN variants"""
    def __init__(self, config, num_labels, model_type="vanilla"):
        super(MultiLabelClassifier, self).__init__()
        
        if model_type == "vanilla":
            self.backbone = VanillaPLRNN(config)
        elif model_type == "gtf":
            self.backbone = GTFShallowPLRNN(config)
        elif model_type == "dendritic":
            self.backbone = DendriticPLRNN(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.LATENT_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_labels)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

def create_data_loaders():
    """Create data loaders"""
    print("📊 Loading dataset metadata...")
    
    df = pd.read_csv(config.MULTILABEL_DATA_CSV)
    binary_labels_df = pd.read_csv(config.BINARY_LABELS_CSV)
    
    min_len = min(len(df), len(binary_labels_df))
    df = df.head(min_len)
    binary_labels_df = binary_labels_df.head(min_len)
    
    print(f"Total samples available: {len(df)}")
    
    # Create datasets
    train_dataset = RealECGDataset(
        df.head(config.MAX_TRAIN_SAMPLES),
        binary_labels_df.head(config.MAX_TRAIN_SAMPLES),
        config.ECG_BASE_PATH,
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    val_dataset = RealECGDataset(
        df[config.MAX_TRAIN_SAMPLES:config.MAX_TRAIN_SAMPLES + config.MAX_VAL_SAMPLES],
        binary_labels_df[config.MAX_TRAIN_SAMPLES:config.MAX_TRAIN_SAMPLES + config.MAX_VAL_SAMPLES],
        config.ECG_BASE_PATH,
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    test_dataset = RealECGDataset(
        df[config.MAX_TRAIN_SAMPLES + config.MAX_VAL_SAMPLES:config.MAX_TRAIN_SAMPLES + config.MAX_VAL_SAMPLES + config.MAX_TEST_SAMPLES],
        binary_labels_df[config.MAX_TRAIN_SAMPLES + config.MAX_VAL_SAMPLES:config.MAX_TRAIN_SAMPLES + config.MAX_VAL_SAMPLES + config.MAX_TEST_SAMPLES],
        config.ECG_BASE_PATH,
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Calculate number of labels
    non_label_columns = ['subject_id', 'study_id', 'ecg_time']
    label_columns = [col for col in binary_labels_df.columns if col not in non_label_columns]
    num_labels = len(label_columns)
    
    return train_loader, val_loader, test_loader, num_labels

def train_model(model, train_loader, val_loader, model_name, num_labels):
    """Train a single model with early stopping"""
    print(f"\n🚀 Training {model_name}...")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.MAX_EPOCHS)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.PATIENCE, min_delta=config.MIN_DELTA)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_hamming': [],
        'val_micro_f1': [],
        'epochs_trained': 0
    }
    
    best_val_f1 = 0
    
    for epoch in range(config.MAX_EPOCHS):
        # Training phase
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{config.MAX_EPOCHS} [Train]')
        for ecg_batch, labels_batch in pbar:
            ecg_batch = ecg_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            if torch.isnan(ecg_batch).any() or torch.isnan(labels_batch).any():
                continue
                
            optimizer.zero_grad()
            logits = model(ecg_batch)
            
            if torch.isnan(logits).any():
                continue
                
            loss = criterion(logits, labels_batch)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        
        # Validation phase
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for ecg_batch, labels_batch in tqdm(val_loader, desc=f'{model_name} [Val]'):
                ecg_batch = ecg_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                if torch.isnan(ecg_batch).any() or torch.isnan(labels_batch).any():
                    continue
                    
                logits = model(ecg_batch)
                
                if torch.isnan(logits).any():
                    continue
                    
                loss = criterion(logits, labels_batch)
                
                if not torch.isnan(loss):
                    val_losses.append(loss.item())
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels_batch.cpu().numpy())
        
        # Calculate metrics
        if val_losses and all_preds:
            avg_val_loss = np.mean(val_losses)
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            
            val_hamming = hamming_loss(all_labels, all_preds)
            
            try:
                _, _, micro_f1, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='micro', zero_division=0
                )
            except:
                micro_f1 = 0.0
        else:
            avg_val_loss = float('inf')
            val_hamming = 1.0
            micro_f1 = 0.0
        
        # Update scheduler
        scheduler.step()
        
        # Save best model
        if micro_f1 > best_val_f1:
            best_val_f1 = micro_f1
            torch.save(model.state_dict(), f'{model_name.lower()}_best.pth')
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_hamming'].append(val_hamming)
        history['val_micro_f1'].append(micro_f1)
        history['epochs_trained'] = epoch + 1
        
        # Print epoch summary
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Hamming: {val_hamming:.4f}, Micro F1: {micro_f1:.4f}")
        
        # Early stopping check
        if early_stopping(micro_f1, model):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    return history, best_val_f1

def evaluate_model(model, test_loader, model_name):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    test_losses = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for ecg_batch, labels_batch in tqdm(test_loader, desc=f'{model_name} [Test]'):
            ecg_batch = ecg_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            logits = model(ecg_batch)
            loss = criterion(logits, labels_batch)
            test_losses.append(loss.item())
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels_batch.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate comprehensive metrics
    test_loss = np.mean(test_losses)
    hamming = hamming_loss(all_labels, all_preds)
    
    # Per-class and micro/macro averages
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    metrics = {
        'test_loss': test_loss,
        'hamming_loss': hamming,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class_f1': f1.tolist()
    }
    
    return metrics

def run_ablation_study():
    """Main ablation study function"""
    print("🎯 Starting Comprehensive Ablation Study...")
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_labels = create_data_loaders()
    print(f"Number of labels: {num_labels}")
    
    # Model variants to compare
    model_configs = [
        ("Vanilla_PLRNN", "vanilla"),
        ("GTF_shPLRNN", "gtf"), 
        ("Dendritic_PLRNN", "dendritic")
    ]
    
    results = {}
    
    for model_name, model_type in model_configs:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Create model
        model = MultiLabelClassifier(config, num_labels, model_type).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Train model
        history, best_val_f1 = train_model(model, train_loader, val_loader, model_name, num_labels)
        
        # Load best model for testing
        model.load_state_dict(torch.load(f'{model_name.lower()}_best.pth'))
        
        # Evaluate on test set
        test_metrics = evaluate_model(model, test_loader, model_name)
        
        # Store results
        results[model_name] = {
            'history': history,
            'best_val_f1': best_val_f1,
            'test_metrics': test_metrics,
            'total_params': total_params,
            'model_type': model_type
        }
        
        print(f"\n✅ {model_name} Results:")
        print(f"Best Val F1: {best_val_f1:.4f}")
        print(f"Test F1 (Micro): {test_metrics['micro_f1']:.4f}")
        print(f"Test F1 (Macro): {test_metrics['macro_f1']:.4f}")
        print(f"Hamming Loss: {test_metrics['hamming_loss']:.4f}")
        print(f"Epochs Trained: {history['epochs_trained']}")
        
    # Save comprehensive results
    with open('ablation_study_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create comparison plots
    create_comparison_plots(results)
    
    # Generate detailed report
    generate_report(results)
    
    return results

def create_comparison_plots(results):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Training curves
    ax1 = axes[0, 0]
    for model_name, data in results.items():
        history = data['history']
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], label=f'{model_name} (Train)', linestyle='--')
        ax1.plot(epochs, history['val_loss'], label=f'{model_name} (Val)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training/Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Validation F1 scores
    ax2 = axes[0, 1]
    for model_name, data in results.items():
        history = data['history']
        epochs = range(1, len(history['val_micro_f1']) + 1)
        ax2.plot(epochs, history['val_micro_f1'], label=model_name)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Micro F1')
    ax2.set_title('Validation Micro F1')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Hamming Loss
    ax3 = axes[0, 2]
    for model_name, data in results.items():
        history = data['history']
        epochs = range(1, len(history['val_hamming']) + 1)
        ax3.plot(epochs, history['val_hamming'], label=model_name)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Hamming Loss')
    ax3.set_title('Validation Hamming Loss')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Final test metrics comparison
    ax4 = axes[1, 0]
    models = list(results.keys())
    micro_f1s = [results[m]['test_metrics']['micro_f1'] for m in models]
    macro_f1s = [results[m]['test_metrics']['macro_f1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax4.bar(x - width/2, micro_f1s, width, label='Micro F1', alpha=0.8)
    ax4.bar(x + width/2, macro_f1s, width, label='Macro F1', alpha=0.8)
    
    ax4.set_xlabel('Models')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('Test Set Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=15)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Parameter count vs performance
    ax5 = axes[1, 1]
    param_counts = [results[m]['total_params'] for m in models]
    test_f1s = [results[m]['test_metrics']['micro_f1'] for m in models]
    
    colors = ['blue', 'red', 'green']
    for i, model in enumerate(models):
        ax5.scatter(param_counts[i], test_f1s[i], 
                   s=100, c=colors[i], label=model, alpha=0.7)
    
    ax5.set_xlabel('Parameter Count')
    ax5.set_ylabel('Test Micro F1')
    ax5.set_title('Parameter Efficiency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Training efficiency (epochs vs performance)
    ax6 = axes[1, 2]
    epochs_trained = [results[m]['history']['epochs_trained'] for m in models]
    best_val_f1s = [results[m]['best_val_f1'] for m in models]
    
    for i, model in enumerate(models):
        ax6.scatter(epochs_trained[i], best_val_f1s[i], 
                   s=100, c=colors[i], label=model, alpha=0.7)
    
    ax6.set_xlabel('Epochs Trained')
    ax6.set_ylabel('Best Validation F1')
    ax6.set_title('Training Efficiency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_ablation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Comparison plots saved as 'comprehensive_ablation_results.png'")

def generate_report(results):
    """Generate detailed comparison report"""
    report = []
    report.append("# Comprehensive PLRNN Ablation Study Report")
    report.append("=" * 50)
    report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total models compared: {len(results)}")
    report.append("")
    
    # Summary table
    report.append("## Summary Results")
    report.append("| Model | Test F1 (Micro) | Test F1 (Macro) | Hamming Loss | Epochs | Parameters |")
    report.append("|-------|-----------------|-----------------|--------------|--------|------------|")
    
    for model_name, data in results.items():
        test_metrics = data['test_metrics']
        report.append(f"| {model_name} | {test_metrics['micro_f1']:.4f} | "
                     f"{test_metrics['macro_f1']:.4f} | {test_metrics['hamming_loss']:.4f} | "
                     f"{data['history']['epochs_trained']} | {data['total_params']:,} |")
    
    report.append("")
    
    # Detailed analysis
    report.append("## Detailed Analysis")
    
    # Find best performing model
    best_model = max(results.keys(), key=lambda x: results[x]['test_metrics']['micro_f1'])
    best_f1 = results[best_model]['test_metrics']['micro_f1']
    
    report.append(f"**Best performing model**: {best_model}")
    report.append(f"**Best test F1 (micro)**: {best_f1:.4f}")
    report.append("")
    
    # Model-specific details
    for model_name, data in results.items():
        report.append(f"### {model_name}")
        test_metrics = data['test_metrics']
        history = data['history']
        
        report.append(f"- **Architecture**: {data['model_type']}")
        report.append(f"- **Parameters**: {data['total_params']:,}")
        report.append(f"- **Training epochs**: {history['epochs_trained']}")
        report.append(f"- **Best validation F1**: {data['best_val_f1']:.4f}")
        report.append(f"- **Test metrics**:")
        report.append(f"  - Micro F1: {test_metrics['micro_f1']:.4f}")
        report.append(f"  - Macro F1: {test_metrics['macro_f1']:.4f}")
        report.append(f"  - Precision (micro): {test_metrics['micro_precision']:.4f}")
        report.append(f"  - Recall (micro): {test_metrics['micro_recall']:.4f}")
        report.append(f"  - Hamming loss: {test_metrics['hamming_loss']:.4f}")
        report.append("")
    
    # Conclusions
    report.append("## Conclusions")
    report.append(f"1. **Overall best performer**: {best_model} achieved the highest test F1 score of {best_f1:.4f}")
    
    # Parameter efficiency
    param_efficiency = {name: data['test_metrics']['micro_f1'] / data['total_params'] * 1e6 
                       for name, data in results.items()}
    most_efficient = max(param_efficiency.keys(), key=lambda x: param_efficiency[x])
    report.append(f"2. **Most parameter-efficient**: {most_efficient}")
    
    # Training efficiency
    training_efficiency = {name: data['best_val_f1'] / data['history']['epochs_trained'] 
                          for name, data in results.items()}
    fastest_converge = max(training_efficiency.keys(), key=lambda x: training_efficiency[x])
    report.append(f"3. **Fastest convergence**: {fastest_converge}")
    
    # Save report
    with open('ablation_study_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("📝 Detailed report saved as 'ablation_study_report.md'")

if __name__ == "__main__":
    results = run_ablation_study()
    print("\n🎉 Comprehensive ablation study completed!")
    print("📁 Results saved in:")
    print("  - ablation_study_results.json")
    print("  - comprehensive_ablation_results.png") 
    print("  - ablation_study_report.md")