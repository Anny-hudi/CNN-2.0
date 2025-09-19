# CNN 模型参数与层级设定

本文档总结了 5 天、20 天、60 天输入下的 CNN 模型结构及参数设定。

---

## 📌 1. 输入数据
- **5-day image**: 32 × 15  
- **20-day image**: 64 × 60  
- **60-day image**: 96 × 180  

---

## 📌 2. 模型结构

### 🔹 5-day 模型
- **输入尺寸**: 32 × 15  
- **卷积模块**:
  1. Conv: 5×3 kernel, 64 filters, ReLU  
     MaxPool: 2×1  
  2. Conv: 5×3 kernel, 128 filters, ReLU  
     MaxPool: 2×1  
- **全连接层 (FC)**: 15360  
- **输出层**: Softmax  

---

### 🔹 20-day 模型
- **输入尺寸**: 64 × 60  
- **卷积模块**:
  1. Conv: 5×3 kernel, 64 filters, ReLU  
     MaxPool: 2×1  
  2. Conv: 5×3 kernel, 128 filters, ReLU  
     MaxPool: 2×1  
  3. Conv: 5×3 kernel, 256 filters, ReLU  
     MaxPool: 2×1  
- **全连接层 (FC)**: 46080  
- **输出层**: Softmax  

---

### 🔹 60-day 模型
- **输入尺寸**: 96 × 180  
- **卷积模块**:
  1. Conv: 5×3 kernel, 64 filters, ReLU  
     MaxPool: 2×1  
  2. Conv: 5×3 kernel, 128 filters, ReLU  
     MaxPool: 2×1  
  3. Conv: 5×3 kernel, 256 filters, ReLU  
     MaxPool: 2×1  
  4. Conv: 5×3 kernel, 512 filters, ReLU  
     MaxPool: 2×1  
- **全连接层 (FC)**: 184320  
- **输出层**: Softmax  

---

## 📌 3. 说明
- 每个 `conv` 使用 5×3 卷积核，激活函数为 ReLU。  
- 每个 `MaxPool` 使用 2×1 池化操作。  
- 最终一个 CNN 模块的输出展平成一维向量，进入全连接层 (FC)。  
- 最终通过 **Softmax 层** 输出分类概率（“上”和“下”）。  

---
