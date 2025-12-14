import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. 复用任务二的二分类数据（保证数据一致性）
iris = load_iris()
y_bin = np.where(iris.target == 0, 0, 1)
X_3d = iris.data[:, [0, 1, 2]]
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length']

# 数据标准化
scaler = StandardScaler()
X_3d_scaled = scaler.fit_transform(X_3d)

# 训练模型（用全量数据，保证概率分布完整性）
clf_3d = LogisticRegression(max_iter=500, random_state=42)
clf_3d.fit(X_3d_scaled, y_bin)

# 2. 生成3D网格（用于预测概率）
# 控制网格密度（15×15×15），平衡效果与速度
x1 = np.linspace(X_3d_scaled[:, 0].min() - 1, X_3d_scaled[:, 0].max() + 1, 15)
x2 = np.linspace(X_3d_scaled[:, 1].min() - 1, X_3d_scaled[:, 1].max() + 1, 15)
x3 = np.linspace(X_3d_scaled[:, 2].min() - 1, X_3d_scaled[:, 2].max() + 1, 15)
xx, yy, zz = np.meshgrid(x1, x2, x3)

# 预测每个网格点的概率（属于Class 1的概率）
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
probs = clf_3d.predict_proba(grid_points)[:, 1].reshape(xx.shape)

# 3. 绘制3D概率图（曲面+等高线，完全匹配PPT样式）
fig = plt.figure(figsize=(14, 12), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# 核心：3D概率曲面（颜色映射概率，冷暖色板PPT常用）
surf = ax.plot_surface(
    xx[:, :, 10], yy[:, :, 10], probs[:, :, 10] * 5,  # 固定x3中间层，概率缩放为高度
    facecolors=plt.cm.coolwarm(probs[:, :, 10]),  # 颜色=概率
    alpha=0.8, edgecolor='none'
)

# 底面等高线（补充2D视角，PPT必备元素）
ax.contourf(
    xx[:, :, 0], yy[:, :, 0], probs[:, :, 0],
    zdir='z', offset=X_3d_scaled[:, 2].min() - 2,  # 等高线置于Z轴下方
    cmap='coolwarm', alpha=0.5
)

# 叠加原始数据点（区分两类）
mask_0 = y_bin == 0
mask_1 = y_bin == 1
ax.scatter(
    X_3d_scaled[mask_0, 0], X_3d_scaled[mask_0, 1], X_3d_scaled[mask_0, 2],
    c='red', s=80, edgecolors='black', label='Setosa (Class 0)', zorder=10
)
ax.scatter(
    X_3d_scaled[mask_1, 0], X_3d_scaled[mask_1, 1], X_3d_scaled[mask_1, 2],
    c='blue', s=80, edgecolors='black', label='Others (Class 1)', zorder=10
)

# 样式调整
ax.view_init(elev=25, azim=50)  # 优化视角，突出概率渐变
ax.set_xlabel(feature_names[0], fontsize=12)
ax.set_ylabel(feature_names[1], fontsize=12)
ax.set_zlabel('Probability (Scaled)', fontsize=12)
ax.set_title('Task 3: 3D Probability Map (Class 1 Probability)', fontsize=16)
ax.legend(loc='upper right', fontsize=11)

# 添加概率颜色条
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=ax, pad=0.1, shrink=0.7)
cbar.set_label('Probability of Class 1 (Others)', fontsize=11)

# 保存图片
plt.tight_layout()
plt.savefig("task3_3d_probability_map.png", dpi=300, bbox_inches='tight')
plt.close()

print("任务三完成！已保存3D概率图：task3_3d_probability_map.png")
print("可视化包含：3D概率曲面 + 底面等高线 + 原始数据点，完全匹配PPT样式")