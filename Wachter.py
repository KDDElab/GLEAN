import os
import torch
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from carla.data import DataCatalog
from carla.models import MLModelCatalog
from carla.recourse_methods import Wachter
from carla.data.pipelining import fit_scaler, fit_encoder, scale, encode, decode, descale
import torch.nn as nn

class CustomData(DataCatalog):
    def __init__(
        self,
        data_name: str,
        data: pd.DataFrame,
        target_column: str,
        continuous_columns: list,
        categorical_columns: list = [],
        immutables_columns: list = [],
        test_size: float = 0.2,
        random_state: int = 42,
        scaling_method: str = "MinMax",
        encoding_method: str = "Identity",
    ):
        self.name = data_name
        self._target = target_column
        self._continuous = continuous_columns
        self._categorical = categorical_columns
        self._immutables = immutables_columns
        
        data[target_column] = data[target_column].astype(int)
        assert set(data[target_column].unique()) == {0, 1}, "目标变量必须是二分类（0 和 1）！"
        
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            data.drop(columns=[target_column]),
            data[target_column],
            test_size=test_size,
            stratify=data[target_column],
            random_state=random_state
        )
        
        self._df_train = pd.concat([self._X_train, self._y_train], axis=1)
        self._df_test = pd.concat([self._X_test, self._y_test], axis=1)
        self._df = data
        
        # 关键修复：即使没有分类特征，也初始化一个虚拟编码器
        self.scaler = fit_scaler(scaling_method, self._X_train)
        if not categorical_columns:
            # 使用空编码器防止后续操作报错
            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.encoder.fit(pd.DataFrame(columns=categorical_columns))  # 拟合空数据
            self._identity_encoding = True
        else:
            self.encoder = fit_encoder(encoding_method, self._X_train[self.categorical])
            self._identity_encoding = (encoding_method == "Identity")

    @property
    def categorical(self):
        return self._categorical

    @property
    def continuous(self):
        return self._continuous

    @property
    def immutables(self):
        return self._immutables

    @property
    def target(self):
        return self._target

    @property
    def df(self):
        return self._df.copy()

    @property
    def df_train(self):
        return self._df_train.copy()

    @property
    def df_test(self):
        return self._df_test.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.drop(columns=[self.target]).copy()
        output = scale(self.scaler, self.continuous, output)
        if not self._identity_encoding and self.encoder is not None:
            output = encode(self.encoder, self.categorical, output)
        return output

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        if not self._identity_encoding and self.encoder is not None:
            output = decode(self.encoder, self.categorical, output)
        output = descale(self.scaler, self.continuous, output)
        return output

# 加载数据
data = pd.read_csv('data1/26_optdigits.csv')
X = data.drop('class', axis=1)
y = data['class']

distance_list = []
sparsity_count_list = []  
sparsity_sum_list = []    

normal = X[y == 0]
centroid = np.mean(normal, axis=0)

# 初始化数据目录
data_catalog = CustomData(
    data_name="optdigits",
    data=data,
    target_column='class',
    continuous_columns=data.drop('class', axis=1).columns.tolist(),
    categorical_columns=[],  # 显式指定无分类特征
    immutables_columns=[],
    test_size=0.2,
    random_state=42,
    scaling_method="MinMax",
    encoding_method="Identity"
)

# 初始化模型
num_features = len(data_catalog.continuous)
carla_model = MLModelCatalog(
    data=data_catalog,
    model_type="ann",
    layers=[num_features, 50, 2],  # 二分类输出层为1节点
    backend="pytorch",
    optimizer="adam",
    batch_size=32,
    epochs=10,
    criterion=nn.CrossEntropyLoss(),  # 二分类专用损失函数
    load_online=False,
)

# 训练模型
carla_model.train(learning_rate=0.001, epochs=10)

test_positive_samples = data_catalog.df_test[data_catalog.df_test['class'] == 1]

proximity_list = []

for i in range(len(test_positive_samples)):
    sample = test_positive_samples.iloc[[i]]   # 包含 'class' 列
    processed_sample = data_catalog.transform(sample).astype(np.float32)  # 标准化后的输入
    print(f"处理样本 {i+1}/{X[y == 1].shape[0]}")
    

    # 初始化 Wachter（适配 CARLA 0.0.5 参数）
    hyperparams = {
        "loss_type": "BCE",       # 必须为 BCE 或 MSE
        "y_target": torch.tensor([1.0], dtype=torch.float32),       # 二分类目标需为 one-hot 格式
        "lr": 0.01,               # 学习率
        "lambda_": 0.1,           # 正则化参数
        "n_iter": 1000,           # 最大迭代次数
    }

    wachter = Wachter(
        mlmodel=carla_model,
        hyperparams=hyperparams
    )
    
    cf = wachter.get_counterfactuals(factuals=processed_sample)
    cf_original_scale = data_catalog.inverse_transform(cf)
    proximity = np.linalg.norm(cf_original_scale.values[0] - sample.drop('class', axis=1).values[0].astype(np.float32))
    print("cf_original_scale.values", cf_original_scale.values[0])
    print("sample.drop('class', axis=1).values", sample.drop('class', axis=1).values[0].astype(np.float32))
    print("proximity", proximity)
    proximity_list.append(proximity)
    
    for _, cf_row in cf_original_scale.iterrows():
        # 转换为 numpy 数组以便计算距离
        cf_array = cf_row.values.reshape(1, -1)
        sample_array = cf_original_scale.values[0]
        
        diff = np.abs(cf_array[0] - sample.drop('class', axis=1).values[0].astype(np.float32))
        
        # 1. 变化特征数（非零差异的特征数量）
        # 可设置阈值（如0.1）判断是否为有效变化，此处以非零为判断标准
        sparsity_count = np.sum(diff > 0.5)  # 避免浮点精度误差
        
        # 2. 总变化幅度（L1范数）
        sparsity_sum = np.sum(diff)
        
        # 计算接近度（原始样本与反事实的距离）
        # proximity = np.linalg.norm(sample_array - cf_array)
        
        # 计算反事实与正常样本中心的距离
        distance = np.linalg.norm(cf_array[0] - centroid.values.reshape(1, -1)[0])
        
        sparsity_count_list.append(sparsity_count)
        sparsity_sum_list.append(sparsity_sum)
        
        # proximity_list.append(proximity)
        distance_list.append(distance)

print("proximity_list: ", np.mean(proximity_list))
print("平均接近度:", np.mean(proximity_list))
print("平均距离:", np.mean(distance_list))
print("平均变化特征数:", np.mean(sparsity_count_list))
print("平均总变化幅度:", np.mean(sparsity_sum_list))