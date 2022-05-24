import torchvision.datasets as datasets
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import time
from torchvision.transforms.functional import to_pil_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 경로 지정
path2data = './data'
os.makedirs(path2data, exist_ok=True)

# Transformation 정의
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# MNIST dataset 불러오기
train_ds = datasets.MNIST(path2data, train=True, transform=train_transform, download=True)

# 샘플 이미지 확인하기
img, label = train_ds.data, train_ds.targets

# 차원 추가
if len(img.shape) == 3:
    img = img.unsqueeze(1)  # B*C*H*W

# 그리드 생성
img_grid = utils.make_grid(img[:40], nrow=8, padding=2)


def show(img):
    npimg = img.numpy()
    npimg_tr = npimg.transpose((1, 2, 0))  # [C,H,W] -> [H,W,C]
    plt.imshow(npimg_tr, interpolation='nearest')
    plt.show()


# 데이터 로더 생성
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

# 파라미터 설정
params = {'num_classes': 10,
          'nz': 100,
          'input_size': (1, 28, 28)}


# Generator: 가짜 이미지를 생성합니다.
# noise와 label을 결합하여 학습합니다..

class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.num_classes = params['num_classes']  # 클래스 수, 10
        self.nz = params['nz']  # 노이즈 수, 100
        self.input_size = params['input_size']  # (1,28,28)

        self.gen = nn.Sequential(
            nn.Linear(self.nz, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, int(np.prod(self.input_size))),
            nn.Tanh()
        )

    def forward(self, noise):
        # noise와 label 결합
        # print("noise.size(): ",noise.size()) #torch.Size([batch_size, 100])
        # print("labels.size(): ", labels.size()) # torch.Size([batch_size])
        # print("self.label_emb(labels).size(): ", self.label_emb(labels).size()) #torch.Size([batch_size, 10])
        # print("gen_input.size(): ", gen_input.size()) #torch.Size([batch_size, 110])
        x = self.gen(noise)
        # print("x.size(): ", x.size()) #torch.Size([batch_size, 784])
        x = x.view(x.size(0), *self.input_size)
        # print("x.size(): ", x.size()) #torch.Size([batch_size, 1, 28, 28])
        return x
model_gen = Generator(params).to(device)
"""
# check
x = torch.randn(16, 100, device=device)  # 노이즈
label = torch.randint(0, 10, (16,), device=device)  # 레이블
model_gen = Generator(params).to(device)
out_gen = model_gen(x, label)  # 가짜 이미지 생성
print("out_gen.shape: ",out_gen.shape) # torch.Size([16, 1, 28, 28])
"""

# Discriminator: 가짜 이미지와 진짜 이미지를 식별합니다.
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.input_size = params['input_size']
        self.num_classes = params['num_classes']

        self.dis = nn.Sequential(
            nn.Linear(int(np.prod(self.input_size)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # 이미지와 label 결합
        # print("img.size(): ", img.size()) #torch.Size([batch_size, 1, 28, 28])
        # print("labels.size(): ", labels.size()) #torch.Size([batch_size])
        image = img.view(img.size(0), -1)
        # print("image.size(): ", image.size()) #torch.Size([batch_size, 784])
        # print("label_embedding.size(): ", label_embedding.size()) #torch.Size([batch_size, 10])
        dis_input = image

        # print("dis_input.size(): ", dis_input.size()) #torch.Size([batch_size, 794])
        x = self.dis(dis_input)
        # print("x.size(): ", x.size()) #torch.Size([batch_size, 1])
        return x
model_dis = Discriminator(params).to(device)
"""
# check
x = torch.randn(16, 1, 28, 28, device=device)
label = torch.randint(0, 10, (16,), device=device)
model_dis = Discriminator(params).to(device)
out_dis = model_dis(x, label)
print("out_dis.shape: ",out_dis.shape) # torch.Size([16, 1])
"""

# 가중치 초기화
def initialize_weights(model):
    classname = model.__class__.__name__
    # fc layer
    if classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    # batchnorm
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


# 가중치 초기화 적용
model_gen.apply(initialize_weights);
model_dis.apply(initialize_weights);

# 손실 함수
loss_func = nn.BCELoss()

from torch import optim

lr = 2e-4
beta1 = 0.5
beta2 = 0.999

# optimization
opt_dis = optim.Adam(model_dis.parameters(), lr=lr, betas=(beta1, beta2))
opt_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=(beta1, beta2))

nz = params['nz']
print("nz: ", nz)
num_epochs = 100

loss_history = {'gen': [],
                'dis': []}

# 학습
batch_count = 0
start_time = time.time()
model_dis.train()
model_gen.train()

for epoch in range(num_epochs):
    for xb, yb in train_dl:
        ba_si = xb.shape[0]
        # print("xb.size(): ", xb.size()) #torch.Size([batch_size, 1, 28, 28])
        # print("yb.size(): ", yb.size()) #torch.Size([batch_size])
        # print("ba_si: ",ba_si) # ba_si:  64
        xb = xb.to(device)
        yb = yb.to(device)

        noise = torch.randn(ba_si, 100).to(device)  # 노이즈 생성, mean 0, variance 1 from a normal distribution, out~N(0,1)
        # print("noise.size()",noise.size()) #torch.Size([batch_size, 100])

        yb_real = torch.Tensor(ba_si, 1).fill_(1.0).to(device)  # real_label
        yb_fake = torch.Tensor(ba_si, 1).fill_(0.0).to(device)  # fake_label
        # print("yb_real.size(): ", yb_real.size()) # torch.Size([64, 1]), fill with tensor([[1.],...,[1.]], device='cuda:0')
        # print("yb_fake.size(): ", yb_fake.size()) # torch.Size([64, 1]), fill with tensor([[0.],...,[0.]], device='cuda:0')



        # Discriminator
        model_dis.zero_grad()

        # 진짜 이미지 판별
        out_dis = model_dis(xb)
        loss_real = loss_func(out_dis, yb_real)

        # 가짜 이미지 생성
        out_gen = model_gen(noise)

        # 가짜 이미지 판별
        out_dis = model_dis(out_gen.detach())
        loss_fake = loss_func(out_dis, yb_fake)

        loss_dis = (loss_real + loss_fake) / 2
        loss_dis.backward()
        opt_dis.step()

        # Genetator
        model_gen.zero_grad()

        # 가짜 이미지 판별
        out_dis = model_dis(out_gen)

        loss_gen = loss_func(out_dis, yb_real)
        loss_gen.backward()
        opt_gen.step()

        loss_history['gen'].append(loss_gen.item())
        loss_history['dis'].append(loss_dis.item())

        batch_count += 1
        if batch_count % 1000 == 0:
            print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' % (
                epoch, loss_gen.item(), loss_dis.item(), (time.time() - start_time) / 60))

# plot loss history
plt.figure(figsize=(10, 5))
plt.title('Loss Progress')
plt.plot(loss_history['gen'], label='Gen. Loss')
plt.plot(loss_history['dis'], label='Dis. Loss')
plt.xlabel('batch count')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 가중치 저장
path2models = './models/'
os.makedirs(path2models, exist_ok=True)
path2weights_gen = os.path.join(path2models, 'weights_gen.pt')
path2weights_dis = os.path.join(path2models, 'weights_dis.pt')

torch.save(model_gen.state_dict(), path2weights_gen)
torch.save(model_dis.state_dict(), path2weights_dis)

# 가중치 불러오기
weights = torch.load(path2weights_gen)
model_gen.load_state_dict(weights)

# evalutaion mode
model_gen.eval()

# fake image 생성
with torch.no_grad():
    fixed_noise = torch.randn(16, 100, device=device)
    label = torch.randint(0, 10, (16,), device=device)
    img_fake = model_gen(fixed_noise).detach().cpu()
print(img_fake.shape)

# 가짜 이미지 시각화
plt.figure(figsize=(10, 10))
for ii in range(16):
    plt.subplot(4, 4, ii + 1)
    plt.imshow(to_pil_image(0.5 * img_fake[ii] + 0.5), cmap='gray')
    plt.axis('off')
plt.show()

# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    show(img_grid)

