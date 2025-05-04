import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import logging
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import timm
import time
import datetime
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Remove hidden files starting with ._
def remove_hidden_files(directory):
    directory = Path(directory)
    count = 0
    for class_dir in ['positive', 'negative']:
        class_path = directory / class_dir
        if class_path.exists():
            for file in class_path.glob("._*"):
                file.unlink()
                count += 1
    if count > 0:
        logging.info(f"Removed {count} hidden files starting with ._")

# Image dataset class
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['positive', 'negative']  # positive=0, negative=1
        
        self.images = []
        self.labels = []
        
        # Load all image paths and labels
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise ValueError(f"Directory does not exist: {class_dir}")
                
            for img_path in class_dir.glob("*.[jJ][pP][gG]"):
                self.images.append(str(img_path))
                self.labels.append(class_idx)
                
        logging.info(f"Found {len(self.images)} images")
        logging.info(f"Positive images (0): {self.labels.count(0)}")
        logging.info(f"Negative images (1): {self.labels.count(1)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logging.error(f"Failed to load image {img_path}: {e}")
            # Return the first image in the dataset as a fallback
            return self.__getitem__(0)

# Improved ViT model - optimized for M1 but with increased accuracy
class ImprovedViTModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super().__init__()
        # Use pretrained ViT model - choose a more powerful model for better accuracy
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
        
        # Get feature dimension
        in_features = self.backbone.head.in_features
        
        # Replace classification head with a more complex structure for better expressiveness
        self.backbone.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# Train model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=5):
    best_val_acc = 0.0
    best_model_path = 'best_vit_model_m1.pth'
    
    # Record total training start time
    total_start_time = time.time()
    
    # Early stopping mechanism
    no_improve_epochs = 0
    
    # Record metrics during training
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logging.info(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc='Training')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        epoch_train_loss = train_loss/train_total
        epoch_train_acc = train_correct/train_total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validation')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{val_loss/val_total:.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        epoch_val_loss = val_loss/val_total
        val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step(epoch_val_loss)  # Use validation loss to adjust learning rate
        
        # Calculate current epoch time
        epoch_time = time.time() - epoch_start_time
        # Calculate remaining time
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining_time = epoch_time * remaining_epochs
        
        # Format time display
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
        
        logging.info(f'Epoch {epoch+1} completed: Train Accuracy={100.*epoch_train_acc:.2f}%, '
                     f'Validation Accuracy={100.*val_acc:.2f}%')
        logging.info(f'Epoch time: {epoch_time_str}, Estimated remaining time: {remaining_time_str}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'Saved best model with validation accuracy: {100.*val_acc:.2f}%')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            logging.info(f'Validation accuracy not improved, {no_improve_epochs} epochs without improvement')
            
            # Early stopping
            if no_improve_epochs >= patience:
                logging.info(f'Early stopping: No improvement for {patience} epochs')
                break
    
    # Calculate total training time
    total_training_time = time.time() - total_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_training_time)))
    logging.info(f'Training completed! Total time: {total_time_str}')
    
    # Plot accuracy changes during training
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_accuracy.png')
    logging.info("Training accuracy chart saved as training_accuracy.png")
    
    # Plot loss chart separately
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_loss.png')
    logging.info("Training loss chart saved as training_loss.png")
    
    return best_model_path, train_accuracies, val_accuracies, train_losses, val_losses

# Evaluate model
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    # For confusion matrix calculation
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing')
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            test_bar.set_postfix({
                'loss': f'{test_loss/test_total:.4f}',
                'acc': f'{100.*test_correct/test_total:.2f}%'
            })
    
    # Calculate accuracy
    accuracy = test_correct / test_total
    
    # Calculate precision, recall and F1 score
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print evaluation results
    logging.info(f"\nTest Set Evaluation Results:")
    logging.info(f"Accuracy: {100.*accuracy:.2f}%")
    logging.info(f"Precision: {100.*precision:.2f}%")
    logging.info(f"Recall: {100.*recall:.2f}%")
    logging.info(f"F1 Score: {100.*f1:.2f}%")
    
    # Display confusion matrix as a table
    classes = ['Positive', 'Negative']
    logging.info(f"Confusion Matrix:")
    
    # Print header
    header = f"{'Actual\\Predicted':<15}"
    for cls in classes:
        header += f"{cls:^10}"
    logging.info(header)
    
    # Print table content
    for i, cls in enumerate(classes):
        row = f"{cls:<15}"
        for j in range(len(classes)):
            row += f"{cm[i, j]:^10}"
        logging.info(row)
    
    return accuracy, precision, recall, f1

def main():
    set_seed()
    
    # Configuration parameters - settings for improved accuracy
    data_dir = 'training_dataset'
    batch_size = 12  # Reduce batch size to fit M1 memory
    num_epochs = 30  # Increase training epochs
    learning_rate = 5e-6  # Lower learning rate for better convergence
    weight_decay = 0.02  # Increase weight decay to reduce overfitting
    
    # Remove hidden files
    remove_hidden_files(data_dir)
    
    # Data preprocessing - use stronger data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    full_dataset = ImageDataset(data_dir, transform=train_transform)
    
    # Split into training, validation and test sets (70% train, 10% validation, 20% test)
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Set different transforms for validation and test sets
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform
    
    logging.info(f"Dataset split: Training {train_size} images, Validation {val_size} images, Testing {test_size} images")
    
    # Create data loaders - optimized for M1 - fix multiprocessing issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Change to 0 to avoid multiprocessing serialization issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,  # Change to 0 to avoid multiprocessing serialization issues
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,  # Change to 0 to avoid multiprocessing serialization issues
        pin_memory=True
    )
    
    # Set device - optimized for M1
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create model
    model = ImprovedViTModel(dropout_rate=0.25)  # Increase dropout to reduce overfitting
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing for better generalization
    
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Use ReduceLROnPlateau scheduler to adjust learning rate based on validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Train model
    best_model_path, train_accs, val_accs, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, 
        optimizer, scheduler, num_epochs, device,
        patience=8  # Set early stopping patience
    )
    
    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate model on test set
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, criterion, device)
    
    # Rename best model
    final_model_path = 'improved_vit_sentiment_model.pth'
    os.rename(best_model_path, final_model_path)
    logging.info(f"Model training completed, saved as: {final_model_path}")
    
    # Save training metrics
    metrics = {
        'train_accuracies': train_accs,
        'val_accuracies': val_accs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1
    }
    
    # Save metrics as numpy arrays
    np.save('training_metrics.npy', metrics)
    logging.info("Training metrics saved as training_metrics.npy")
    
    return final_model_path

if __name__ == "__main__":
    main() 