import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import sys
import os
# 获取项目根目录（EMOTION_SYSTEM 文件夹路径）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import FERDataset
from models.face_model import FaceEmotionModel
from utils.transforms import train_transform, inference_transform
from sklearn.metrics import classification_report

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 分别创建训练集（含增强）和测试集（不增强）
    train_ds = FERDataset("data/FER2013/fer2013.csv", transform=train_transform)
    test_ds = FERDataset("data/FER2013/fer2013.csv", transform=inference_transform)

    # 使用固定种子划分，保证训练/测试不重叠
    train_size = int(0.8 * len(train_ds))
    test_size = len(train_ds) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(train_ds, [train_size, test_size], generator=generator)

    generator = torch.Generator().manual_seed(42)
    _, test_dataset = random_split(test_ds, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = FaceEmotionModel().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 学习率调度：当验证准确率停滞 3 个 epoch 时自动降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    num_epochs = 30
    best_accuracy = 0.0

    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        train_losses.append(epoch_loss)

        # ===== 测试准确率 =====

        model.eval()

        correct = 0
        total = 0

        y_true = []
        y_pred = []

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total

        test_accuracies.append(accuracy)

        # 调整学习率
        scheduler.step(accuracy)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} Loss:{epoch_loss:.4f} Accuracy:{accuracy:.2f}% LR:{current_lr:.6f}")

        # 保存最优模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "models/face_emotion_model.pth")
            print(f"  ✓ 保存最优模型 (Accuracy: {accuracy:.2f}%)")

    print(f"\n训练结束，最佳准确率: {best_accuracy:.2f}%")

    # ===== Loss 曲线 =====
    plt.figure()
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png")
    plt.close()

    # ===== Accuracy 曲线 =====
    plt.figure()
    plt.plot(test_accuracies)
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("accuracy_curve.png")
    plt.close()

    # ===== 混淆矩阵 =====
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    print(classification_report(y_true, y_pred))
    print("训练完成，图像已保存")