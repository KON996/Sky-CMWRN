import torch
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)
    print(f'保存当前最优模型参数到文件: {path}')

def plot_losses(train_losses, val_losses, title):
    """绘制训练和验证损失曲线"""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'train_result/{title.lower().replace(" ", "_")}_curve.png')
    plt.close()

def plot_accuracies(train_accuracies, val_accuracies):
    epochs = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_result/rain_train_accuracy_curve.png')
    plt.close()