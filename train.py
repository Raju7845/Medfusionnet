import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MedFusionNet
from dataset import MedMultimodalDataset

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        images = batch['image'].to(device)
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        task_ids = batch['task_id']

        optimizer.zero_grad()
        outputs = model(images, ids, mask)
        
        # Calculate loss based on task_id
        loss = 0
        for i, tid in enumerate(task_ids):
            task_key = ['bc', 'cc', 'pcos'][tid]
            loss += criterion(outputs[task_key][i].unsqueeze(0), labels[i].unsqueeze(0))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedFusionNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Placeholder for actual data loading logic
    # In a real run, you would populate dummy_data with paths from config['paths']
    dummy_data = [] 
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = MedMultimodalDataset(dummy_data, transform=transform)
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # STAGE 1: Freeze Backbones
    for param in model.img_encoder.parameters(): param.requires_grad = False
    for param in model.text_encoder.parameters(): param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate_init'])
    
    print("Starting Stage 1: Backbone Frozen")
    for epoch in range(config['training']['stage1_epochs']):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")

    # STAGE 2: Full Fine-tuning
    for param in model.parameters(): param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate_ft'])
    
    print("Starting Stage 2: End-to-End Fine-tuning")
    best_loss = float('inf')
    for epoch in range(config['training']['stage2_epochs']):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config['save_path'])
            print(f"Model saved to {config['save_path']}")

if __name__ == "__main__":
    main()