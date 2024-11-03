import torch
import torch.nn as nn
import torch.nn.functional as F

# 옵션 설정 클래스
class Options:
    def __init__(self):
        self.input_nc = 3
        self.output_nc = 3
        self.ndf = 16  # 필터 수를 더 줄여 초경량화
        self.gaussian_kernel_size = 3  # 커널 크기 감소
        self.gaussian_sigma = 0.5  # 가우시안 블러의 시그마 값 감소
        self.lr = 0.0001

# 가중치 초기화 함수
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# 초경량 Blur2BlurModel 정의
class Blur2BlurModel(nn.Module):
    def __init__(self, opt):
        super(Blur2BlurModel, self).__init__()
        self.opt = opt
        self.conv1 = nn.Conv2d(opt.input_nc, opt.ndf, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(opt.ndf, opt.output_nc, kernel_size=3, padding=1)

        # 가우시안 커널 생성 및 Conv2d에 적용
        kernel = create_gaussian_kernel(opt.gaussian_kernel_size, opt.gaussian_sigma)
        self.gaussian_blur = nn.Conv2d(3, 3, kernel_size=opt.gaussian_kernel_size, padding=opt.gaussian_kernel_size // 2, groups=3, bias=False)
        self.gaussian_blur.weight.data = kernel.expand(3, 1, -1, -1)
        self.gaussian_blur.weight.requires_grad = False

    def forward(self, x):
        x = self.gaussian_blur(x)  # 가우시안 블러 적용
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# 초경량 NAFNet 정의
class NAFNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(NAFNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)  # 필터 수 더 줄임
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Blur2Blur -> NAFNet 통합 모델 정의
class Blur2BlurNAFNet(nn.Module):
    def __init__(self, opt):
        super(Blur2BlurNAFNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blur2blur = Blur2BlurModel(opt).to(self.device)
        self.nafenet = NAFNet().to(self.device)

    def forward(self, x):
        blur2blur_output = self.blur2blur(x)
        final_output = self.nafenet(blur2blur_output)
        return final_output

# 가우시안 커널 생성 함수
def create_gaussian_kernel(kernel_size, sigma):
    """2D Gaussian Kernel 생성"""
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2
    y = torch.arange(kernel_size) - (kernel_size - 1) / 2
    x_grid, y_grid = torch.meshgrid(x, y)
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel
