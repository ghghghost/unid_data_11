import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model2 import Options, Blur2BlurNAFNet # 모델 및 옵션, 초기화 함수 가져오기

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# 학습 함수 정의
def train(model, dataloader, num_epochs, opt):
    criterion = nn.L1Loss()  # L1 손실 함수 사용
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            real_A = data['input'].to(device)
            real_B = data['target'].to(device)
            output = model(real_A)
            loss = criterion(output, real_B)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
            optimizer.step()

            epoch_loss += loss.item()
        scheduler.step()  # 스케줄러로 학습률 감소
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "blur2blur_nafenet.pth")
    print("Model saved as blur2blur_nafenet.pth")

# 데이터셋 정의
class CustomDataset(Dataset):
    def __init__(self, inputs_dir, targets_dir, transform=None):
        """디렉토리 경로를 받아 CustomDataset 초기화"""
        self.inputs = sorted([os.path.join(inputs_dir, f) for f in os.listdir(inputs_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.targets = sorted([os.path.join(targets_dir, f) for f in os.listdir(targets_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform or transforms.ToTensor()  # 기본 변환을 ToTensor로 설정

    def __len__(self):
        return len(self.inputs)  # 데이터셋의 크기 반환

    def __getitem__(self, idx):
        """인덱스로 특정 데이터를 로드하고 반환"""
        input_image = Image.open(self.inputs[idx]).convert("RGB")  # PIL 이미지로 로드
        target_image = Image.open(self.targets[idx]).convert("RGB")  # PIL 이미지로 로드
        
        # 변환 적용
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return {'input': input_image, 'target': target_image}

# 데이터 경로 및 변환 설정
inputs_dir = '/content/drive/MyDrive/Uni-D-Datathon-4th/data/Training/clean'
targets_dir = '/content/drive/MyDrive/Uni-D-Datathon-4th/data/Training/noisy'
transform = transforms.ToTensor()

# 데이터셋 및 DataLoader 설정
dataset = CustomDataset(inputs_dir, targets_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 옵션 설정 및 모델 초기화
opt = Options()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Blur2BlurNAFNet(opt).to(device)
model.apply(weights_init)  # 가중치 초기화 적용

# 학습 실행
num_epochs = 6
train(model, dataloader, num_epochs, opt)
