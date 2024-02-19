#!/usr/bin/env python
# coding: utf-8

# ### 导入相关的包

# In[1]:


from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LambdaLR,ReduceLROnPlateau
from torchvision import datasets
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
from sklearn.neighbors import NearestNeighbors
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
import random


# ### 部署模型，修改分类层

# In[2]:


# 使用预训练参数
from models.SqueezeNet import SqueezeNet

model = SqueezeNet(version="1_1")
model.load_state_dict(torch.load('squeezenet1_1_weights.pth'))


# 冻结参数层
for param in model.parameters():
    param.requires_grad = False

# 修改最后一层

print(model.classifier[1])
final_conv = nn.Conv2d(512, 128, kernel_size=1)
init.normal_(final_conv.weight, mean=0.0, std=0.01)
init.constant_(final_conv.bias, 0.0)
model.classifier[1] = final_conv


model = model.to('cuda')



for params in model.classifier[1].parameters():
    params.requires_grad = True
    
# 解冻模型的最后1个Fire模块
for param in model.features[-1:].parameters():
    param.requires_grad = True

print(model)


# In[3]:


# test_img_001_0 = Image.open('./data_cropped/001/001_0.bmp')
# test_img_001_1 = Image.open('./data_cropped/001/001_1.bmp')
# test_img_001_2 = Image.open('./data_cropped/001/001_2.bmp')
# test_img_003_3 = Image.open('./data_cropped/003/003_3.bmp')
# test_img_007_3 = Image.open('./data_cropped/007/007_3.bmp')
# test_img_003_1 = Image.open('./data_cropped/003/003_1.bmp')
# test_img_003_2 = Image.open('./data_cropped/003/003_2.bmp')

# lables = ["001_0", "001_1", "001_2", "003_3", "007_3", "003_1", "003_2"]

transforming = transforms.Compose([
    transforms.Resize((224, 224)),
    # 数据增强
    transforms.RandomHorizontalFlip(),  # 随机水平翻转，有助于人脸识别任务
    transforms.RandomRotation(10),  # 随机旋转，-10到10度之间
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机调整亮度、对比度和饱和度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 随机仿射变换
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
])

# aligned = []
# aligned.append(transforming(test_img_001_0))
# aligned.append(transforming(test_img_001_1))
# aligned.append(transforming(test_img_001_2))
# aligned.append(transforming(test_img_003_3))
# aligned.append(transforming(test_img_007_3))
# aligned.append(transforming(test_img_003_1))
# aligned.append(transforming(test_img_003_2))

# model.eval()
# aligned = torch.stack(aligned).to('cuda')
# emdeddings = model(aligned).cpu().detach()
 
# print(emdeddings[0].shape)
# dists = [[(e1 - e2).norm().item() for e2 in emdeddings] for e1 in emdeddings]
# pd.DataFrame(dists, columns=lables, index=lables)




# ### 读取并且定制数据集

# In[4]:


loss = TripletMarginLoss(margin=0.2, p=2)


# 分割三元组数据集
class TripletFaceDataset(Dataset):
    def __init__(self, image_folder,persons,transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.persons = persons

    def __getitem__(self, _):
        anchor_person = random.choice(self.persons)
        positive_person = anchor_person
        negative_person = random.choice(self.persons)
        while negative_person == anchor_person:
            negative_person = random.choice(self.persons)

        anchor_img_path = random.choice(os.listdir(
            os.path.join(self.image_folder, anchor_person)))
        positive_img_path = random.choice(os.listdir(
            os.path.join(self.image_folder, positive_person)))
        negative_img_path = random.choice(os.listdir(
            os.path.join(self.image_folder, negative_person)))

        anchor_img = Image.open(os.path.join(
            self.image_folder, anchor_person, anchor_img_path))
        positive_img = Image.open(os.path.join(
            self.image_folder, positive_person, positive_img_path))
        negative_img = Image.open(os.path.join(
            self.image_folder, negative_person, negative_img_path))

        # 进行数据处理
        transform = self.transform

        anchor_img = transform(anchor_img)
        positive_img = transform(positive_img)
        negative_img = transform(negative_img)

        return (anchor_img, positive_img, negative_img),(anchor_person, positive_person, negative_person)

    def __len__(self):
        return len(self.persons) * 5  # 假设每个人有5张图片


# print(labels)

# 切割数据集八二分

batch_size = 64
epochs = 300
workers = 0 if os.name == 'nt' else 8


dataset = datasets.ImageFolder('./data_cropped', transform=transforming)
labels = os.listdir('./data_cropped')
# print(labels)
random.shuffle(labels)
train_idx, test_idx = train_test_split(
    labels, test_size=0.2, random_state=42)


train_dataset = TripletFaceDataset(
    image_folder='./data_cropped', persons=train_idx, transform=transforming
    )

test_dataset = TripletFaceDataset(
    image_folder='./data_cropped', persons=test_idx, transform=transforming
    )

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)



# ### 开始模型训练

# In[7]:


# 定制优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# def lambda_rule(epoch,max_epoch):
#     max_epoch = 20
#     return (1-epoch/max_epoch)**0.9 ##多项式衰减

# scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)


# 定制参考嵌入向量
reference_embeddings = []
reference_labels = []
model.eval()
# 遍历数据集每个文件夹，每个文件夹embeddings计算平均值，放进reference_embeddings,对应的标签放进reference_labels
for label in labels:
    img_path = os.listdir(os.path.join('./data_cropped', label))
    if img_path:
        first_img = img_path[0]
        img = Image.open(os.path.join('./data_cropped', label, first_img))
        img = transforming(img).unsqueeze(0).to('cuda')
        with torch.no_grad():
            embedding = model(img).cpu().detach().numpy()
            temsor_embedding = torch.from_numpy(embedding).to('cuda')
            
        reference_embeddings.append(temsor_embedding)
        reference_labels.append(label)



# ### 开始训练

# In[8]:


writer = SummaryWriter('runs/triplet_loss_experiment')


print('Start Training')
print('-'*10)


for epoch in range(epochs):
    print('\n循环 {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)
    
    model.train()
    total_running_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        anchor, positive, negative = data[0]
        anchor, positive, negative = anchor.to('cuda'), positive.to('cuda'), negative.to('cuda')
        optimizer.zero_grad()
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        loss_val = loss(anchor_out, positive_out, negative_out)
        loss_val.backward()
        optimizer.step()
        total_running_loss += loss_val.item()
        running_loss += loss_val.item()
        # writer.iteration += 1
        if i % 10 == 9:
            writer.add_scalar('loss', running_loss / 10, epoch * len(train_loader) + i)
            running_loss = 0.0
        avg_loss = total_running_loss / len(train_loader)
        writer.add_scalar('avg_loss', avg_loss, epoch)
    print('Epoch{} average loss:{}'.format(epoch, avg_loss))
    print('Epoch{} finished'.format(epoch))
    
    
    model.eval()
    ## we use acc to evaluate the model
    with torch.no_grad():
        # 用标签来评估模型
        total = 0
        correct = 0
        reference_embeddings_np = np.vstack([embedding.cpu().numpy() for embedding in reference_embeddings])
        near_nn = NearestNeighbors(n_neighbors=1, algorithm='auto',metric='euclidean',).fit(reference_embeddings_np)
        # use labels to evaluate the model
        for i,data in enumerate(test_loader,0):
                anchor_validation, _, _ = data[0]
                anchor_label, _, _ = data[1]
                batch_sizes = anchor_validation.size(0)
                total += batch_sizes
                anchor_validation = anchor_validation.to('cuda')
                # print(anchor.shape)
                anchor_out_validation = model(anchor_validation).cpu().detach().numpy()
                # print('anchor_out_validation:', anchor_out_validation)
                distances, indices = near_nn.kneighbors(anchor_out_validation)
                indices = indices.reshape(-1)
                # print('indices:', indices)
                # print('distances:', distances)
                for j in range(batch_sizes):
                    # print('anchor_label:', anchor_label[j])
                    # print('reference_labels[indices[j]]:', reference_labels[indices[j]])
                    if reference_labels[indices[j]] == anchor_label[j]:
                        correct += 1
                
        acc = correct / total
        print('---------------------------------------------------------------------------')
        print('acc:', acc)
        print('correct:', correct)
        
        writer.add_scalar('acc', acc, epoch)
        
        
        # 用scheduler来更新学习率
        scheduler.step(acc)
        current_lr = scheduler.get_last_lr()
        print("Current learning rate: ", current_lr)

        
print('Finished Training')

writer.close()

            

