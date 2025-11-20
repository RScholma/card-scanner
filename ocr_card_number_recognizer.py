import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import re
import random

# -----------------------------
# 1. Charset and utility functions
# -----------------------------
charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
char_to_idx = {c: i for i, c in enumerate(charset)}
idx_to_char = {i: c for i, c in enumerate(charset)}

def encode_label(label):
    return [char_to_idx[c] for c in label]

def decode_label(indices):
    return ''.join([idx_to_char[i] for i in indices if i < len(charset)])

# -----------------------------
# 1b. Formatting OCR output
# -----------------------------
def format_card_number(raw_output: str) -> str:
    raw_output = raw_output.strip()
    match = re.match(r"([A-Z]+(?:-[A-Z]+)?)(\d+)", raw_output, re.IGNORECASE)
    if not match:
        return raw_output
    prefix, number = match.groups()
    return f"{prefix}{number.zfill(3)}"

# -----------------------------
# 2. Dataset class
# -----------------------------
class CardNumberDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_height=32, transform=None):
        self.images_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.labels_dir = labels_dir
        self.img_height = img_height
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        label_path = os.path.join(self.labels_dir, os.path.basename(img_path).replace(".png", ".txt"))

        # Load image
        image = Image.open(img_path).convert("L")  # grayscale
        w, h = image.size
        new_w = int(self.img_height * w / h)
        image = image.resize((new_w, self.img_height), Image.BILINEAR)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Load label
        with open(label_path, 'r') as f:
            label_str = f.read().strip()
        label = torch.tensor(encode_label(label_str), dtype=torch.long)
        return image, label

# -----------------------------
# 3. CRNN model definition
# -----------------------------
class CRNN(nn.Module):
    def __init__(self, img_channels=1, num_classes=len(charset)+1):  # +1 for CTC blank
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256,512,3,1,1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512,512,3,1,1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512,512,2,1,0), nn.ReLU()
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, num_classes)  # 512 = bidirectional hidden size doubled

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "Height must be 1 after CNN layers"
        conv = conv.squeeze(2).permute(0,2,1)
        rnn_out,_ = self.rnn(conv)
        output = self.fc(rnn_out)
        output = output.permute(1,0,2)  # [seq_len, batch, num_classes] for CTC
        return output

# -----------------------------
# 4. Collate with padding for variable width
# -----------------------------
def collate_fn(batch):
    imgs, labels = zip(*batch)
    max_w = max(img.size(2) for img in imgs)
    padded_imgs = []
    for img in imgs:
        pad_w = max_w - img.size(2)
        padded_imgs.append(torch.nn.functional.pad(img, (0,pad_w)))
    imgs_tensor = torch.stack(padded_imgs, 0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_tensor = torch.cat(labels)
    return imgs_tensor, labels_tensor, label_lengths
'''
# -----------------------------
# 5. Training loop
# -----------------------------
def train_crnn(model, dataset, val_dataset=None, epochs=1, batch_size=4, lr=1e-3, device='cpu'):
    model.to(device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    criterion = nn.CTCLoss(blank=len(charset))  # last index = blank
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels, label_lengths in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            input_lengths = torch.full((imgs.size(0),), outputs.size(0), dtype=torch.long).to(device)
            loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(train_loader):.4f}")

        if val_dataset:
            val_acc = evaluate_crnn(model, val_dataset, device)
            print(f"Validation accuracy: {val_acc:.2f}%")
'''
# -----------------------------
# 5b. Training loop with quick prediction check
# -----------------------------
def train_crnn(model, dataset, val_dataset=None, epochs=1, batch_size=4, lr=1e-3, device='cpu'):
    model.to(device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    criterion = nn.CTCLoss(blank=len(charset))  # last index = blank
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels, label_lengths in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            input_lengths = torch.full((imgs.size(0),), outputs.size(0), dtype=torch.long).to(device)
            loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(train_loader):.4f}")

        # Quick random prediction check on 3 random samples
        model.eval()
        sample_indices = random.sample(range(len(dataset)), min(3, len(dataset)))
        for i, idx in enumerate(sample_indices):
            img, label = dataset[idx]  # dataset[idx] will re-apply transforms each time
            raw_pred = predict_raw(model, img, device)
            formatted_pred = format_card_number(raw_pred)
            filename = os.path.basename(dataset.images_paths[idx])
            print(f"Random Sample '{filename}': Raw: '{raw_pred}' | Formatted: '{formatted_pred}'")
        model.train()

        # Optional validation
        if val_dataset:
            val_acc = evaluate_crnn(model, val_dataset, device)
            print(f"Validation accuracy: {val_acc:.2f}%")

# -----------------------------
# 5c. raw prediction function
# -----------------------------
def predict_raw(model, img, device='cpu'):
    """Return raw string before formatting"""
    model.eval()
    with torch.no_grad():
        img = img if isinstance(img, Image.Image) else transforms.ToPILImage()(img.cpu())
        img = img.convert("L")
        h = 32
        w = int(h * img.width / img.height)
        img = img.resize((w,h), Image.BILINEAR)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        output = model(img_tensor)
        output = output.permute(1,0,2)
        probs = output.softmax(2)
        _, indices = probs.max(2)
        indices = indices[0].cpu().numpy()
        pred = []
        prev = -1
        for i in indices:
            if i != prev and i != len(charset):
                pred.append(i)
            prev = i
        return decode_label(pred)


# -----------------------------
# 6. Inference
# -----------------------------
def predict(model, img, device='cpu'):
    model.eval()
    with torch.no_grad():
        img = img.convert("L")
        h = 32
        w = int(h * img.width / img.height)
        img = img.resize((w,h), Image.BILINEAR)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        output = model(img_tensor)
        output = output.permute(1,0,2)
        probs = output.softmax(2)
        _, indices = probs.max(2)
        indices = indices[0].cpu().numpy()
        pred = []
        prev = -1
        for i in indices:
            if i != prev and i != len(charset):
                pred.append(i)
            prev = i
        pred_str = decode_label(pred)
        return format_card_number(pred_str)

# -----------------------------
# 7. Evaluation
# -----------------------------
def evaluate_crnn(model, dataset, device='cpu'):
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    correct, total = 0, 0
    for imgs, labels, label_lengths in loader:
        imgs = imgs.to(device)
        preds = [predict(model, transforms.ToPILImage()(img.cpu()), device) for img in imgs]
        start = 0
        for i, length in enumerate(label_lengths):
            target = decode_label(labels[start:start+length].cpu().numpy())
            start += length
            if preds[i] == target:
                correct += 1
            total += 1
    return correct / total * 100

# -----------------------------
# 8. Main execution
# -----------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Paths — replace with your own
    images_dir = "dataset/card_number_crops/train/images"
    labels_dir = "dataset/card_number_crops/train/labels"

    # Quick check if dataset exists
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError("Dataset folders not found. Please check paths.")

    # -----------------------------
    # Data augmentation
    # -----------------------------
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    # Create datasets
    dataset = CardNumberDataset(images_dir, labels_dir,transform=train_transform)
    print(f"Loaded {len(dataset)} images.")

    if len(dataset) == 0:
        raise ValueError("No images found in dataset. Check your paths and file extensions.")

    # Instantiate model
    model = CRNN()

    # Train for 50 epochs
    print("Starting training for 50 epochs")
    train_crnn(model, dataset, epochs=50, batch_size=4, device=device)

    # Test prediction on first image
    test_img_path = dataset.images_paths[0]
    test_img = Image.open(test_img_path)

    raw_pred = predict_raw(model, test_img, device=device)
    formatted_pred = format_card_number(raw_pred)

    print(f"Prediction for {os.path.basename(test_img_path)}:")
    print(f"  Raw output      : '{raw_pred}'")
    print(f"  Formatted output: '{formatted_pred}'")

    # Optionally evaluate on whole dataset
    acc = evaluate_crnn(model, dataset, device=device)
    print(f"Dataset accuracy (quick check): {acc:.2f}%")
