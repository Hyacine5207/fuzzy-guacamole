import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. 数据准备（二分类+3个特征，匹配PPT任务二）
iris = load_iris()
# 二分类转换：Setosa（0）vs Versicolor+Virginica（1）
y_bin = np.where(iris.target == 0, 0, 1)
# 选择3个特征（萼片长度、萼片宽度、花瓣长度，PPT指定维度）
X_3d = iris.data[:, [0, 1, 2]]
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length']

# 数据标准化（提升模型拟合效果，PPT隐含要求）
scaler = StandardScaler()
X_3d_scaled = scaler.fit_transform(X_3d)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X_3d_scaled, y_bin, test_size=0.3, random_state=42
)

# 2. 训练逻辑回归模型（拟合3D决策超平面）
clf_3d = LogisticRegression(max_iter=500, random_state=42)
clf_3d.fit(X_train, y_train)
test_acc = clf_3d.score(X_test, y_test)

# 3. 计算3D决策超平面（方程：w1x1 + w2x2 + w3x3 + b = 0）
w = clf_3d.coef_[0]  # 权重向量
b = clf_3d.intercept_[0]  # 偏置项
# 生成x1-x2网格，求解x3（超平面上的点）
x1_min, x1_max = X_3d_scaled[:, 0].min() - 1, X_3d_scaled[:, 0].max() + 1
x2_min, x2_max = X_3d_scaled[:, 1].min() - 1, X_3d_scaled[:, 1].max() + 1
x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 20), np.linspace(x2_min, x2_max, 20))
x3 = -(w[0] * x1 + w[1] * x2 + b) / w[2]

# 4. 绘制3D决策边界（匹配PPT样式）
fig = plt.figure(figsize=(12, 10), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# 绘制决策超平面（半透明蓝色，PPT同款效果）
ax.plot_surface(x1, x2, x3, color='lightblue', alpha=0.5, edgecolor='none')

# 绘制原始数据点（区分两类）
mask_0 = y_bin == 0  # Setosa（红色）
mask_1 = y_bin == 1  # 其他两类（蓝色）
ax.scatter(
    X_3d_scaled[mask_0, 0], X_3d_scaled[mask_0, 1], X_3d_scaled[mask_0, 2],
    c='red', s=80, edgecolors='black', label='Setosa (Class 0)', zorder=5
)
ax.scatter(
    X_3d_scaled[mask_1, 0], X_3d_scaled[mask_1, 1], X_3d_scaled[mask_1, 2],
    c='blue', s=80, edgecolors='black', label='Others (Class 1)', zorder=5
)

# 样式调整（匹配PPT视角和标签）
ax.view_init(elev=20, azim=45)  # 俯仰角20°，方位角45°，PPT默认视角
ax.set_xlabel(feature_names[0], fontsize=12)
ax.set_ylabel(feature_names[1], fontsize=12)
ax.set_zlabel(feature_names[2], fontsize=12)
ax.set_title(f'Task 2: 3D Decision Boundary (Test Acc: {test_acc:.3f})', fontsize=16)
ax.legend(loc='upper right', fontsize=11)
ax.grid(alpha=0.3)

# 保存图片
plt.tight_layout()
plt.savefig("task2_3d_boundary.png", dpi=300, bbox_inches='tight')
plt.close()

print("任务二完成！已保存3D决策边界图：task2_3d_boundary.png")
print(f"二分类准确率：{test_acc:.3f}（Setosa与其他两类线性可分）")