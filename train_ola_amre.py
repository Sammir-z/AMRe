#!/usr/bin/env python3
"""
Ola-AMRe Training Script
Example script for training AMRe with Ola-7b feature extraction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.basic_model import OlaAMReClassifier, OlaFeatureAdapter
from configs.ola_config import OlaConfig, MVSAOlaConfig, AVEOlaConfig, IEMOCAPOlaConfig
from utils.utils import save_checkpoint, load_checkpoint
from utils.metrics import calculate_metrics

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Ola-AMRe Training')
    
    # Dataset and task
    parser.add_argument('--dataset', type=str, default='MVSA', 
                        choices=['MVSA', 'AVE', 'IEMOCAP', 'CREMA-D'],
                        help='Dataset to use')
    
    # Ola settings
    parser.add_argument('--ola_model_path', type=str, default='THUdyh/Ola-7b',
                        help='Path to Ola model')
    parser.add_argument('--ola_feature_dim', type=int, default=768,
                        help='Ola feature dimension')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Model settings
    parser.add_argument('--fusion_method', type=str, default='concat',
                        choices=['concat', 'sum', 'Gate', 'Film', 'CA', 'MMTM', 'CentralNet'],
                        help='Fusion method')
    
    # Other settings
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/ola_amre',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval')
    
    return parser.parse_args()

def get_config_by_dataset(dataset_name):
    """Get configuration based on dataset"""
    if dataset_name == 'MVSA':
        return MVSAOlaConfig.get_args()
    elif dataset_name == 'AVE':
        return AVEOlaConfig.get_args()
    elif dataset_name == 'IEMOCAP':
        return IEMOCAPOlaConfig.get_args()
    elif dataset_name == 'CREMA-D':
        return CREMA_D_OlaConfig.get_args()
    else:
        return OlaConfig.get_args()

def create_model(args):
    """Create Ola-AMRe model"""
    print("Creating Ola-AMRe model...")
    
    # Update args with command line arguments
    args.ola_model_path = getattr(args, 'ola_model_path', 'THUdyh/Ola-7b')
    args.ola_feature_dim = getattr(args, 'ola_feature_dim', 768)
    args.fusion_method = getattr(args, 'fusion_method', 'concat')
    
    # Create model
    model = OlaAMReClassifier(args)
    
    return model

def create_dataloaders(args):
    """Create data loaders (placeholder - implement based on your dataset)"""
    print(f"Creating data loaders for {args.model_name}...")
    
    # TODO: Implement dataset-specific data loaders
    # This is a placeholder - you'll need to implement actual data loading
    # based on your specific datasets
    
    # Example structure:
    # from dataset.dataloader import create_dataloader
    # train_loader = create_dataloader(args, split='train')
    # val_loader = create_dataloader(args, split='val')
    # test_loader = create_dataloader(args, split='test')
    
    # For now, return None placeholders
    print("Warning: Data loaders not implemented. Please implement dataset-specific loaders.")
    return None, None, None

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (datas, labels) in enumerate(dataloader):
        # Move data to device
        if isinstance(datas, (list, tuple)):
            datas = [d.to(device) if d is not None else None for d in datas]
        else:
            datas = datas.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        try:
            outputs = model(datas, labels=labels, epoch=epoch)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                if len(outputs) == 5:  # (fusion_out, out_1, out_2, mask1, mask2)
                    fusion_out = outputs[0]
                elif len(outputs) == 7:  # (fusion_out, out_1, out_2, out_3, mask1, mask2, mask3)
                    fusion_out = outputs[0]
                else:
                    fusion_out = outputs[0]
            else:
                fusion_out = outputs
            
            # Compute loss
            loss = criterion(fusion_out, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(fusion_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')
                
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for datas, labels in dataloader:
            # Move data to device
            if isinstance(datas, (list, tuple)):
                datas = [d.to(device) if d is not None else None for d in datas]
            else:
                datas = datas.to(device)
            labels = labels.to(device)
            
            # Forward pass
            try:
                outputs = model(datas, labels=labels, epoch=-1)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    fusion_out = outputs[0]
                else:
                    fusion_out = outputs
                
                # Compute loss
                loss = criterion(fusion_out, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(fusion_out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            except Exception as e:
                print(f"Error in validation: {e}")
                continue
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def main():
    """Main training function"""
    # Parse arguments
    cmd_args = parse_args()
    
    # Get dataset-specific configuration
    config_args = get_config_by_dataset(cmd_args.dataset)
    
    # Merge command line args with config
    for key, value in vars(cmd_args).items():
        setattr(config_args, key, value)
    
    args = config_args
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create model
    try:
        model = create_model(args)
        model.to(device)
        print(f"Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Please ensure Ola is properly installed and the model path is correct.")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(args)
    
    if train_loader is None:
        print("Data loaders not available. Please implement dataset-specific data loading.")
        print("Example model creation completed successfully.")
        return
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), 
                           lr=args.learning_rate, 
                           weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, f'best_model_{args.dataset}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': args
            }, save_path)
            print(f"New best model saved: {val_acc:.2f}%")
    
    # Test on best model if test loader is available
    if test_loader is not None:
        print("\nTesting on best model...")
        # Load best model
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()