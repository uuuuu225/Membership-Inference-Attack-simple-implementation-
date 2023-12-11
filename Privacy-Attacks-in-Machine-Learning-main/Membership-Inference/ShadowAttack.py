import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import numpy as np
import yaml
import os
from model import PurchaseClassifier,AttackModel
from sklearn.svm import LinearSVC

#导入路径
config_file = './env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']

#读取文件
path_dir = os.path.join(root_dir, 'purchase')
DATASET_PATH = os.path.join(path_dir, 'data')
data_path = os.path.join(DATASET_PATH,'partition')

#读取数据
train_data_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_data.npy'))
train_label_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_label.npy'))
train_data_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_data.npy'))
train_label_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_label.npy'))
train_data = np.load(os.path.join(DATASET_PATH, 'partition', 'train_data.npy'))
train_label = np.load(os.path.join(DATASET_PATH, 'partition', 'train_label.npy'))
test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data.npy'))
test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label.npy'))

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

# 创建目标模型
target_model = PurchaseClassifier(num_classes=100)

# 目标模型训练
target_optimizer = optim.Adam(target_model.parameters(), lr=0.001)
target_criterion = nn.CrossEntropyLoss()
target_epochs = 10
target_train_dataset = MyDataset(train_data, train_label)
target_train_dataloader = DataLoader(target_train_dataset, batch_size=128, shuffle=True)

best_loss = float('inf')  # 损失函数初始化为正无穷大
save_folder = 'best_models'
best_targetmodel_path = os.path.join(save_folder, 'best_target_model.pth')

for epoch in range(target_epochs):
    for data, labels in target_train_dataloader:
        target_optimizer.zero_grad()

        outputs = target_model(data)
        loss = target_criterion(outputs, labels.long())

        loss.backward()
        target_optimizer.step()

        # 计算当前损失
        current_loss = loss.item()

        # 如果当前损失优于之前的最佳损失，保存模型
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': target_model.state_dict(),
                'optimizer_state_dict': target_optimizer.state_dict(),
                'loss': current_loss,
            }, best_targetmodel_path)

    # 每个 epoch 后输出损失函数值
    print(f'Target Model: Epoch [{epoch + 1}/{target_epochs}], Loss: {current_loss}')


# 加载保存的最佳模型
best_checkpoint = torch.load(best_targetmodel_path)
target_model.load_state_dict(best_checkpoint['model_state_dict'])

def add_gaussian_noise(data, noise_std=0.01):
    noise = np.random.normal(0, noise_std, data.shape)
    noisy_data = data + noise
    return np.clip(noisy_data, 0, 1)

# 创建影子模型
shadow_model = PurchaseClassifier(num_classes=100)

# 影子模型训练
shadow_optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)
shadow_criterion = nn.CrossEntropyLoss()
shadow_epochs = 10
shadow_train_data = add_gaussian_noise(train_data)
shadow_train_label = add_gaussian_noise(train_label)
shadow_train_dataset = MyDataset(shadow_train_data, shadow_train_label)
shadow_train_dataloader = DataLoader(shadow_train_dataset, batch_size=128, shuffle=True)

best_loss = float('inf')  # 损失函数，初始化为正无穷大
save_folder = 'best_models'
best_shadowmodel_path = os.path.join(save_folder, 'best_shadow_model.pth')

for epoch in range(shadow_epochs):
    for data, labels in shadow_train_dataloader:
        shadow_optimizer.zero_grad()

        outputs = shadow_model(data)
        loss = shadow_criterion(outputs, labels.long())

        loss.backward()
        shadow_optimizer.step()

        # 计算当前损失
        current_loss = loss.item()

        # 如果当前损失优于之前的最佳损失，保存模型
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': shadow_model.state_dict(),
                'optimizer_state_dict': shadow_optimizer.state_dict(),
                'loss': current_loss,
            }, best_shadowmodel_path)

    # 每个 epoch 后输出损失函数值
    print(f'shadow Model: Epoch [{epoch + 1}/{shadow_epochs}], Loss: {current_loss}')


# 加载保存的最佳模型

best_checkpoint = torch.load(best_shadowmodel_path)
shadow_model.load_state_dict(best_checkpoint['model_state_dict'])


# 攻击模型训练
shadow_test_data = add_gaussian_noise(test_data)
train_data_tr_attack_tensor = torch.Tensor(shadow_train_data)
train_data_te_attack_tensor = torch.Tensor(shadow_test_data)

predict_attack_train1 =  shadow_model(train_data_tr_attack_tensor)
predict_attack_train2 = shadow_model(train_data_te_attack_tensor)

predict_attack_train_member = [("in",d) for d in predict_attack_train1]
predict_attack_train_nonmember = [("out",d) for d in predict_attack_train2]


test_data_tr_attack_tensor = torch.Tensor(train_data)
test_data_te_attack_tensor = torch.Tensor(test_data)

predict_attack_test1 =  target_model(test_data_tr_attack_tensor)
predict_attack_test2 = target_model(test_data_te_attack_tensor)

predict_attack_test_member = [("in",d) for d in predict_attack_test1]
predict_attack_test_nonmember = [("out",d) for d in predict_attack_test2]



X_attack = np.vstack([d[1].detach().numpy() for d in predict_attack_train_member + predict_attack_train_nonmember])
Y_attack = [d[0] for d in predict_attack_train_member + predict_attack_train_nonmember]
X_attack_train = X_attack
X_attack_test = np.vstack([d[1].detach().numpy() for d in predict_attack_test_member + predict_attack_test_nonmember])
Y_attack_train = Y_attack
Y_attack_test = [d[0] for d in predict_attack_test_member + predict_attack_test_nonmember]
#  LinearSVC 模型
#num_trainings = 5
#for _ in range(num_trainings):
 #   classifier = LinearSVC()
  #  classifier.fit(X_attack_train, Y_attack_train)
   # accuracy = classifier.score(X_attack_test, Y_attack_test)
    #print(f"Accuracy after training: {accuracy}")

# 使用 AttackModel 进行训练
input_size = 100  # 指定输入特征的数量
hidden_size = 64   # 指定隐藏层的大小
out_classes = 2    # 指定输出的类别数量

attack_model = AttackModel(input_size, hidden_size, out_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

# 多次训练 AttackModel
num_trainings = 10
for _ in range(num_trainings):
    # 转换为 PyTorch 张量
    X_attack_train_tensor = torch.Tensor(X_attack_train)
    Y_attack_train_tensor = torch.LongTensor([1 if label == 'in' else 0 for label in Y_attack_train])

    # 训练 AttackModel
    optimizer.zero_grad()
    outputs = attack_model(X_attack_train_tensor)
    loss = criterion(outputs, Y_attack_train_tensor)
    loss.backward()
    optimizer.step()

    # 在每次训练后，评估模型的性能
    with torch.no_grad():
        X_attack_test_tensor = torch.Tensor(X_attack_test)
        Y_attack_test_tensor = torch.LongTensor([1 if label == 'in' else 0 for label in Y_attack_test])
        predictions = torch.argmax(attack_model(X_attack_test_tensor), dim=1)
        accuracy = torch.sum(predictions == Y_attack_test_tensor).item() / len(Y_attack_test_tensor)
        print(f"Accuracy after training: {accuracy}")
