import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                           download=True, transform=transform)
# Lấy ảnh đầu tiên và nhãn của nó từ tập train
image, label = train_dataset[0] 

# Vì image đang là Tensor (1, 28, 28), ta cần bỏ chiều channel để vẽ (28, 28)
# Và cần 'un-normalize' (đảo ngược quá trình chuẩn hóa) để ảnh hiện đúng màu
image = image * 0.5 + 0.5 
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Nhãn thực tế: {label}")
plt.show()


