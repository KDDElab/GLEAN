import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from dice_ml import Dice, Data
from dice_ml.model import Model

# 加载数据
data = pd.read_csv('data1/01_yeast.csv')
X = data.drop('class', axis=1)
y = data['class']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 创建 DICE 数据对象
d = Data(
    dataframe=data,
    continuous_features=X.columns.tolist(),
    outcome_name='class'
)

# 初始化模型接口
m = Model(model=model, backend='sklearn')

# 初始化 DICE 解释器
exp = Dice(d, m, method="random")

# 选择一个 class=1 的样本
proximity_list = []
distance_list = []
sparsity_count_list = []  
sparsity_sum_list = []    

normal = X[y == 0]
centroid = np.mean(normal, axis=0)

# 对每个类别为1的样本生成反事实
for i in range(X_test[y == 1].shape[0]):
    sample = X_test[y == 1].iloc[i:i+1]
    print(f"处理样本 {i+1}/{X_test[y == 1].shape[0]}")
    
    # 生成反事实
    dice_exp = exp.generate_counterfactuals(
        query_instances=sample,
        total_CFs=5,  # 生成5个反事实样本
        desired_class=0
    )
    
    # 获取生成的反事实样本
    cf_df = dice_exp.cf_examples_list[0].final_cfs_df
    
    # 确保列顺序一致
    cf_df = cf_df[sample.columns]
    
    proximity = np.linalg.norm(sample.values - cf_df.values)
    proximity_list.append(proximity)
    
    # 计算原始样本与每个反事实样本的距离
    for _, cf_row in cf_df.iterrows():
        # 转换为 numpy 数组以便计算距离
        cf_array = cf_row.values.reshape(1, -1)
        sample_array = sample.values
        
        diff = np.abs(cf_array - sample_array)
        
        # 1. 变化特征数（非零差异的特征数量）
        # 可设置阈值（如0.1）判断是否为有效变化，此处以非零为判断标准
        sparsity_count = np.sum(diff > 1e-8)  # 避免浮点精度误差
        
        # 2. 总变化幅度（L1范数）
        sparsity_sum = np.sum(diff)
        
        # 计算接近度（原始样本与反事实的距离）
        # proximity = np.linalg.norm(sample_array - cf_array)
        
        # 计算反事实与正常样本中心的距离
        distance = np.linalg.norm(cf_array - centroid.values.reshape(1, -1))
        
        sparsity_count_list.append(sparsity_count)
        sparsity_sum_list.append(sparsity_sum)
        
        # proximity_list.append(proximity)
        distance_list.append(distance)

# 计算平均指标
if proximity_list and distance_list:
    print("平均接近度:", np.mean(proximity_list))
    print("平均距离:", np.mean(distance_list))
    print("平均变化特征数:", np.mean(sparsity_count_list))
    print("平均总变化幅度:", np.mean(sparsity_sum_list))
else:
    print("没有生成有效的反事实样本")