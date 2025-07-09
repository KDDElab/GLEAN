import random
import h5py
import numpy as np
import scipy.io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import math
from pyod.models.lof import LOF
import os
import time
from scipy.interpolate import make_interp_spline
from scipy.spatial.distance import pdist, squareform
class Mysuanfa:
    def __init__(self):


        def load_mat_v73(file_path):
            data = {}
            with h5py.File(file_path, 'r') as f:
                # 遍历所有顶层键
                for key in f.keys():
                    # 读取数据集
                    dataset = f[key]
                    if isinstance(dataset, h5py.Dataset):
                        # 转换为 NumPy 数组
                        data[key] = np.array(dataset)
                    elif isinstance(dataset, h5py.Group):
                        # 处理组（如 struct）
                        pass  # 需要递归解析
            return data

        # 加载数据
        mat_data = load_mat_v73('Datasets/wine.mat')
        X = mat_data['X'].T
        y = mat_data['y'].T
        data1 = np.column_stack((X, y))

        # data2 = pd.DataFrame(data1, columns=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26','27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39','40', '41', '42', '43', '44', '45', '46', '47', '48', '49','50', '51', '52', '53', '54', '55', '56', '57', '58', '59','60', '61', '62', '63', '64', 'type'))
        data2 = pd.DataFrame(data1, columns=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 'type'))
        data3 = data2[data2['type'] == 1]
        new_instance = data3.iloc[3].to_frame().T
        self.query_instance_type = new_instance
        self.query_instance_notype = new_instance.drop("type", axis=1)
        qs = data2.loc[data2['type'] == 1]
        df = data2.drop(qs.index, axis=0)
        self.data_full_feature = df
        self.no_type_df = self.data_full_feature.drop("type", axis=1)




         #构建一个LOF异常检测器，同时构建一个可以计算lof得分的函数
        #一个
        self.scaler = StandardScaler()
        data_x = self.scaler.fit_transform(self.no_type_df)
        self.train_data = data_x
        self.model_lof = self.train_lof(self.train_data, 10, 0.1)  # 50表示近邻，0.05表示期待出现异常的比例

        #一个
        self.clf = LOF(n_neighbors=10, contamination=0.1)
        self.clf.fit(self.train_data)





    def show_picture(self):
        if not os.path.exists(self.figure_save_path):
            os.makedirs(self.figure_save_path)
        filename = '方向损失' + str(time.time()) + '.png'
        self.picture.savefig(os.path.join(self.figure_save_path, filename))
        self.picture.show()

    def generate_counterfactuals(self, query_instance, total_CFs, initialization="kdtree",
                                 desired_range=None, desired_class="opposite", proximity_weight=0.2,
                                 sparsity_weight=0.2, diversity_weight=5.0, categorical_penalty=0.1,
                                 algorithm="DiverseCF", features_to_vary="all", permitted_range=None,
                                 yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist",
                                 feature_weights="inverse_mad", stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                                 posthoc_sparsity_algorithm="binary", maxiterations=500, thresh=1e-2,
                                 verbose=False):
        self.population_size = 10 * total_CFs
        self.total_CFs = total_CFs

        test_pred = self.clf.decision_function(self.biaozhunhua(query_instance))
        # score = self.clf.decision_scores_
        self.test_pred = test_pred
        self.y_test_pred = self.clf.predict(self.biaozhunhua(query_instance))





        desired_class = 1 - self.y_test_pred
        self.class_d = desired_class
        self.do_param_initializations()
        features_to_vary = self.no_type_df.columns.tolist()
        self.continuous_feature_names = self.no_type_df.columns.tolist()

        self.find_counterfactuals(query_instance, desired_range, desired_class, features_to_vary,
                                  maxiterations=500, thresh=1e-7, verbose=False)

    def paint_picture(self, query_instance, itration):
        for i in range(len(query_instance)):
            self.picture.scatter(query_instance[i][0], query_instance[i][1], color='purple')
            self.picture.text(query_instance[i][0] + 0.1, query_instance[i][1] + 0.1, itration)
            break

    def biaozhunhua(self, queryinstance):
        shuju = queryinstance
        if not isinstance(queryinstance, pd.DataFrame):
            shuju = pd.DataFrame(shuju)
        shuju.columns = self.no_type_df.columns.tolist()
        shuju_plus = self.scaler.transform(shuju)
        return shuju_plus

    def do_param_initializations(self):
        self.feature_range = self.get_valid_feature_range()
        # print('每个特征的上下限')
        # print(self.feature_range)

        # self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights, encoding='label')
        # self.update_hyperparameters(proximity_weight, sparsity_weight, diversity_weight, categorical_penalty)

    def get_valid_feature_range(self):
        stats = self.no_type_df.describe()

        stats_dict = stats.to_dict()

        limits_dict = {}
        for col in self.no_type_df.columns:
            limits_dict[col] = [stats_dict[col]['min'], stats_dict[col]['max']]

        return limits_dict

    def update_hyperparameters(self, proximity_weight, sparsity_weight,
                               diversity_weight, categorical_penalty):
        """Update hyperparameters of the loss function"""

        self.proximity_weight = proximity_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty

    def train_lof(self, train_data, k, contamination):
        lof = LocalOutlierFactor(n_neighbors=k, contamination=contamination, novelty=True)
        lof.fit(train_data)
        return lof

    def perturb(self, lst):
        # 获取列表的行数和列数
        rows, cols = len(lst), len(lst[0])
        # 遍历列表中的每个元素
        for i in range(rows):
            for j in range(cols):
                # 生成随机数
                rand_num = random.uniform(-1, 1)
                # 计算扰动范围
                perturb_range = lst[i][j] * 0.2
                # 计算扰动后的值
                perturbed_val = lst[i][j] + rand_num * perturb_range
                # 更新列表中的元素
                lst[i][j] = perturbed_val
        return lst

    def find_counterfactuals(self, query_instance, desired_range, desired_class,
                             features_to_vary, maxiterations, thresh, verbose):
        """Finds counterfactuals by generating cfs through the genetic algorithm"""

        population = []
        zhong = query_instance.values[0]
        # print(zhong)
        # print(query_instance)
        # exit()
        for i in range(10):
            population.append(list(zhong))

        population = self.perturb(population)
        iterations = 0
        previous_best_loss = -np.inf  # 代表负无穷
        current_best_loss = np.inf  # 代表正无穷

        stop_cnt = 0
        cfs_preds = [np.inf] * self.total_CFs
        to_pred = None
        # self.query_instance_normalized = self.data_interface.normalize_data(self.x1)
        # self.query_instance_normalized = self.query_instance_normalized.astype('float')
        query_instance = zhong
        self.x = []
        self.y = []
        # 绘制问询实例在初始数据集中的位置
        # self.picture.scatter(query_instance[0], query_instance[1], color='green')
        while iterations < maxiterations and self.total_CFs > 0:
            if (abs(previous_best_loss - current_best_loss) <= thresh):
                stop_cnt += 1
            else:
                stop_cnt = 0
            if stop_cnt >= 20:
                break
            previous_best_loss = current_best_loss
            population = np.unique(tuple(map(tuple, population)), axis=0)
            population_fitness = self.compute_loss(population, desired_range, desired_class)
            population_fitness = population_fitness[population_fitness[:, 1].argsort()]

            # a = self.biaozhunhua(population)
            # population_fitness1 = self.compute_loss1(population, desired_range, desired_class)
            # # population_fitness1 = population_fitness1[population_fitness1[:, 1].argsort()]
            # current_best_loss1 = population_fitness1[0]
            # pop = self.compute_loss1(a,desired_range,desired_class)
            # k = self.clf.decision_function(a)

            # pop = np.mean(pop)
            current_best_loss = population_fitness[0][1]

            self.x.append(iterations + 1)
            self.y.append(current_best_loss)

            # to_pred = np.array([population[int(tup[0])] for tup in population_fitness[:self.total_CFs]])

            # if self.total_CFs > 0:
            #     cfs_preds = self._predict_fn_custom(to_pred, desired_class)

            # self.total_CFS of the next generation obtained from the fittest members of current generation
            top_members = self.total_CFs
            new_generation_1 = np.array([population[int(tup[0])] for tup in population_fitness[:top_members]])
            # print('最优父代')
            # print(new_generation_1)
            # self.paint_picture(new_generation_1, iterations)

            # rest of the next generation obtained from top 50% of fittest members of current generation
            rest_members = self.population_size - top_members
            new_generation_2 = None


            # 交叉生成后代
            if rest_members > 0:
                new_generation_2 = np.zeros((rest_members, self.no_type_df.shape[1]))
                for new_gen_idx in range(rest_members):
                    medium = len(population_fitness) / 2
                    parent1 = population[int(random.choice(population_fitness[0:math.floor(medium)])[0])]
                    parent2 = population[int(random.choice(population_fitness[0:math.floor(medium)])[0])]
                    child = self.mate(parent1, parent2, features_to_vary, query_instance)
                    # a1 = self.compute_loss(parent1.reshape(1, 11), [], 0)[0][1]
                    # b1 = self.compute_loss(parent2.reshape(1, 11), [], 0)[0][1]
                    # c1 = self.compute_loss(child.reshape(1, 11), [], 0)[0][1]
                    # if a1 <= b1:
                    #     if a1 <= c1:
                    #         new_generation_2[new_gen_idx] = parent1
                    #     elif c1 <= b1:
                    #         new_generation_2[new_gen_idx] = child
                    #     else:
                    #         new_generation_2[new_gen_idx] = parent2
                    # else:
                    #     if b1 <= c1:
                    #         new_generation_2[new_gen_idx] = parent2
                    #     else:
                    #         new_generation_2[new_gen_idx] = child
                    new_generation_2[new_gen_idx] = child
            # print('杂交子代')
            # print(new_generation_2)
            if new_generation_2 is not None:
                if self.total_CFs > 0:
                    population = np.concatenate([new_generation_1, new_generation_2])
                else:
                    population = new_generation_2
            else:
                raise SystemError("The number of total_Cfs is greater than the population size!")





            # print('##########################')
            iterations += 1
        print('迭代次数')
        print(iterations)
        # self.show_picture()
        self.cfs_preds = []
        self.final_cfs = []
        i = 0
        while i < self.total_CFs:
            q = pd.DataFrame([population[i]])
            predictions = self.clf.predict(self.biaozhunhua(q))
            self.final_cfs.append(population[i])
            # checking if predictions is a float before taking the length as len() works only for array-like
            # elements. isinstance(predictions, (np.floating, float)) checks if it's any float (numpy or otherwise)
            # We do this as we take the argmax if the prediction is a vector -- like the output of a classifier
            if not isinstance(predictions, (np.floating, float)) and len(predictions) > 1:
                self.cfs_preds.append(np.argmax(predictions))
            else:
                self.cfs_preds.append(predictions)
            i += 1

        # converting to dataframe
        cf_instance_df = pd.DataFrame(self.final_cfs, columns=self.no_type_df.columns.tolist())

        cf_instance_df = cf_instance_df.assign(type=self.cfs_preds)
        print(self.query_instance_type)


        for i in range(len(cf_instance_df)):
            if(cf_instance_df.at[i,'type']!=1):
                print('反事实生成成功')
                cf = pd.DataFrame(cf_instance_df.iloc[i]).transpose()
                type_name = self.jurge_type(cf)
                cf_instance_df.at[i,'type'] = type_name

        self.distance_appraise =[]
        self.lof_appraise = []
        self.proximity_list = []
        sparsity_count_list = []
        for i in range(len(cf_instance_df)):
            # 计算原事实到簇心的距离，判断其应属于哪个类
            catilogy = self.compute_belong(self.query_instance_type)
            print('原实例应属于:')
            print(catilogy)
            # 计算反事实到上述簇心的距离，判断哪个算法好
            suppose_heart = self.heart_dictionary[catilogy]
            print(suppose_heart)

            cf = pd.DataFrame(cf_instance_df.iloc[i]).transpose()
            cf = cf.drop('type',axis=1)

            array_1 = np.array(cf.values)
            array_1 = array_1.astype(np.float32)
            array_2 = np.array(suppose_heart.values[0])
            array_2 = array_2.astype(np.float32)
            result = np.linalg.norm(array_1 - array_2, axis=1) 
            modified = [sublist[:-1] for sublist in self.query_instance_type.values]
            print("self.query_instance_type.values[0]", modified)
            

            result11 = np.linalg.norm(array_1 - modified, axis=1)
            diff = np.abs(array_1[0] - modified[0].astype(np.float32))
            sparsity_count = np.sum(diff > 0.01)

            print("result11", result11)
            self.distance_appraise.append(result[0])
            # 计算反事实所处位置的密度，比较哪个算法好
            cf_biaozhunhua = self.biaozhunhua(cf)
            test_pred = self.clf.decision_function(cf_biaozhunhua)
            self.lof_appraise.append(test_pred[0])
            self.proximity_list.append(result11[0])
            sparsity_count_list.append(sparsity_count)



        #输出一则为异常，除非是属于一类
        print(cf_instance_df)
        print('反事实距离得分')
        print(np.mean(self.distance_appraise))
        print('反事实lof得分')
        print(np.mean(self.lof_appraise))
        print('接近度：')
        print(np.mean(self.proximity_list))
        print('特征变化数：')
        print(np.mean(sparsity_count_list))
    def compute_belong(self,queryinstance):
        cluster_centers = np.array(list(self.heart_dictionary.values()))
        cluster_centers = cluster_centers.reshape(cluster_centers.shape[0], -1)
        array_1 = queryinstance.drop("type", axis=1).values[0]
        array_1 = array_1.astype(np.float32)
        array_2 = cluster_centers
        array_2 = array_2.astype(np.float32)
        distances = np.linalg.norm(array_2-array_1 , axis=1)
        nearest_cluster_index = np.argmin(distances)
        nearest_cluster_key = list(self.heart_dictionary.keys())[nearest_cluster_index]
        return nearest_cluster_key

    def jurge_type(self,cf_instance):
        type_name = []
        distance = []
        grouped = self.data_full_feature.groupby('type')
        new_dfs = [grouped.get_group(x) for x in grouped.groups]
        for new_df in new_dfs:
            heart = new_df.drop("type", axis=1).mean().to_frame().transpose()

            array_1 = np.array(cf_instance.drop("type", axis=1).values)
            array_1 = array_1.astype(np.float32)
            array_2 = np.array(heart.values[0])
            array_2 = array_2.astype(np.float32)
            cf_heart_diatance = np.linalg.norm(array_1 - array_2, axis=1)
            distance.append(cf_heart_diatance)
            df = new_df.reset_index(drop=True)
            catilogy = df.at[0,'type']
            type_name.append(catilogy)
        arr = np.array(distance)
        # 获取均值最小的行的索引
        min_index = arr.argmin()
        # 取出均值最小的一维数组
        min_arr = type_name[min_index]
        return min_arr
    def compute_loss(self, cfs, desired_range, desired_class):
        """Computes the overall loss"""

        self.lof_loss = self.computelof_loss(cfs)
        # print('lof值变化：')
        # print(self.lof_loss)

        self.yloss = self.compute_yloss(cfs, desired_range, desired_class)
        # print("y损失")
        # print(self.yloss)

        self.direction_loss = self.compute_direction_loss(cfs)
        # print('方向损失变化')
        # print(self.direction_loss)


        self.proximity_loss = self.compute_proximity_loss(cfs)
        # self.sparsity_loss = self.compute_sparsity_loss(cfs) if self.sparsity_weight > 0 else 0.0
        # 删除了稀疏性度量
        self.distance_loss = self.compute_distance_loss(cfs)

        self.sp_loss = self.compute_proximity_loss(cfs)
        # print('距离损失')
        # print(self.distance_loss)
        # print("#################/n##############")
        # self.loss = np.reshape(np.array(
        #      10*self.lof_loss+self.yloss + self.distance_loss+self.direction_loss),
        #                         (-1, 1))
        self.loss = np.reshape(np.array(
             1*self.yloss + self.distance_loss),
            (-1, 1))
        # 我删除了多样性损失(self.proximity_weight * self.proximity_loss)，self.optics_loss，self.yloss，
        # self.sparsity_weight * self.sparsity_loss
        index = np.reshape(np.arange(len(cfs)), (-1, 1))
        self.loss = np.concatenate([index, self.loss], axis=1)
        return self.loss

    def compute_loss1(self, cfs, desired_range, desired_class):
        """Computes the overall loss"""

        self.lof_loss = self.computelof_loss(cfs)



        return self.lof_loss





    def compute_distance_loss(self, cfs):
        new_cfs = self.biaozhunhua(cfs)
        new_query = self.biaozhunhua(self.query_instance_notype)[0]
        distances = np.linalg.norm(new_cfs - new_query, axis=1)
        return distances

    def compute_direction_loss(self, cfs):
        direct_distance = []
        new_cfs = self.biaozhunhua(cfs)
        cluster_heart, cluster_radius = self.compute_cluster_heart_and_r()
        for i in range(len(cluster_heart)):
            cluster_heart_bianma = self.biaozhunhua(cluster_heart[i])
            c_c_distance = np.linalg.norm(new_cfs -cluster_heart_bianma[0] , axis=1)

            distances = abs(c_c_distance-cluster_radius[i])
            direct_distance.append(distances)

        # 计算每一行的均值
        arr = np.array(direct_distance)
        means = arr.mean(axis=1)

        # 获取均值最小的行的索引
        min_index = means.argmin()

        # 取出均值最小的一维数组
        min_arr = direct_distance[min_index]
        return min_arr

    def compute_cluster_heart_and_r(self):
        self.heart_dictionary = {}
        cluster_heart = []
        cluster_radius = []
        # 使用groupby方法将有相同type值的实例分离出来放到一个新dataframe中
        grouped = self.data_full_feature.groupby('type')
        new_dfs = [grouped.get_group(x) for x in grouped.groups]
        for new_df in new_dfs:
            heart = new_df.drop("type", axis=1).mean().to_frame().transpose()
            cluster_heart.append(heart)
            key = new_df.head(1).iloc[0]['type']
            self.heart_dictionary[key] = heart
            array_1 = self.biaozhunhua(new_df.drop('type',axis=1))
            array_2 = self.biaozhunhua(heart)
            distances = np.linalg.norm(array_1-array_2[0], axis=1)
            r = np.max(distances)
            cluster_radius.append(r)
        return cluster_heart, cluster_radius


    def compute_yloss(self, cfs, desired_range, desired_class):

        arr = desired_class
        cfs = self.biaozhunhua(cfs)
        y_loss = self.clf.decision_function(cfs) - arr
        return y_loss


    def computelof_loss(self, cf_instance):
        df_query = pd.DataFrame(cf_instance)
        df_query.columns = self.no_type_df.columns.tolist()
        orinignal_cf = self.scaler.transform(df_query)

        lof_score = -self.predict_lof(orinignal_cf)  # 此函数表示越小越异常，负数为异常，所以前面加个负号
        return lof_score

    def predict_lof(self, test_data):
        lof_scores = self.model_lof.decision_function(test_data)
        return lof_scores
    def compute_proximity_loss(self, xcf):
        """计算多样性损失"""
        xcf = self.biaozhunhua(xcf)
        distances = pdist(xcf)
        mean = np.mean(distances)
        mean = 1/(1+mean)
        return mean

    def mate(self, k1, k2, features_to_vary, query_instance):
        """这段 Python 代码实现了遗传算法中的交叉（mate）操作，用于生成新的孩子个体。具体来说，它接收两个父亲个体 k1 和 k2，
        以及一些特征名列表 features_to_vary 和查询实例 query_instance。
        它通过将 k1 和 k2 的基因进行混合，生成一个新的孩子个体 one_init，并返回该个体。
        在混合过程中，每个特征会被分别处理，具体操作如下：
        从 k1 和 k2 中获取当前特征的基因值 gp1 和 gp2。
        随机生成一个概率值 prob，如果 prob 小于 0.40，则从 k1 中选择该基因，否则如果 prob 小于 0.80，则从 k2 中选择该基因，
        否则从当前特征的范围中随机选择一个基因，作为新的孩子个体的该特征基因。这样做是为了保证新生代个体的多样性。
        如果当前特征名在 features_to_vary 列表中，则说明该特征是可变的，需要进行基因突变来保持多样性。
        如果该特征是连续的，则从该特征的范围中随机选择一个基因值，否则从该特征的取值范围中随机选择一个基因值。
        返回新的孩子个体 one_init。
        总之，这个 mate 函数是遗传算法中的一个关键部分，通过混合两个父亲的基因，生成新的孩子个体，同时加入了基因突变来保证后代个体的多样性，
        从而实现了优良个体的筛选和多样性的维护。"""
        # chromosome for offspring全零数组存储孩子个体
        one_init = np.zeros(self.no_type_df.shape[1])
        for j in range(self.no_type_df.shape[1]):
            gp1 = k1[j]
            gp2 = k2[j]
            feat_name = self.no_type_df.columns.tolist()[j]
            new_gp1 = np.random.uniform(0.8 * gp1, 1.2 * gp1)
            new_gp2 = np.random.uniform(0.8 * gp2, 1.2 * gp2)

            # random probability
            prob = random.random()

            if prob < 0.4:
                # if prob is less than 0.40, insert gene from parent 1
                one_init[j] = new_gp1
            elif prob < 0.8:
                # if prob is between 0.40 and 0.80, insert gene from parent 2
                one_init[j] = new_gp2
            else:

                # otherwise insert random gene(mutate) for maintaining diversity
                if feat_name in features_to_vary:
                    if feat_name in self.continuous_feature_names:
                        one_init[j] = np.random.uniform(self.feature_range[feat_name][0],
                                                        self.feature_range[feat_name][1])
                    else:
                        one_init[j] = np.random.choice(self.feature_range[feat_name])
                else:
                    one_init[j] = query_instance[j]
        return one_init
        # a1 = self.compute_loss(k1.reshape(1,11),[],0)[0][1]
        # b1 = self.compute_loss(k2.reshape(1, 11), [], 0)[0][1]
        # c1 = self.compute_loss(one_init.reshape(1, 11), [], 0)[0][1]
        #
        # if a1 <= b1:
        #     if a1 <= c1:
        #         return k1
        #     elif c1 <= b1:
        #         return one_init
        #     else:
        #         return k2
        # else:
        #     if b1 <= c1:
        #         return k2
        #     else:
        #         return one_init
        # return one_init

# start_time = time.time()
# pd.set_option('display.max_columns', 1000)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth', 1000)
mysuanfa = Mysuanfa()
mysuanfa.generate_counterfactuals(mysuanfa.query_instance_notype, 20)
# end_time = time.time()
# print("时间差为"+str(end_time - start_time))
# spline = make_interp_spline(mysuanfa.x,mysuanfa.y,k=3)
# x_smooth = np.linspace(min(mysuanfa.x),max(mysuanfa.x),200)
# plt.plot(x_smooth,spline(x_smooth),marker = 'o',markersize=4,linestyle='-',linewidth=1)
# plt.xlabel("The number of iterations")
# plt.ylabel("Loss function")
#
# plt.title("GAI")
# plt.show()
