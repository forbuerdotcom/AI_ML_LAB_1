import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("ЧАСТЬ 1. Исследование датасета load_digits")
print("=" * 60)

digits = datasets.load_digits()
data = scale(digits.data)  
labels = digits.target  

n_samples, n_features = data.shape
n_unique = len(np.unique(labels))

print(f"1. Размерность данных: {data.shape}")
print(f"   Количество объектов: {n_samples}")
print(f"   Количество признаков: {n_features}")
print(f"   Количество уникальных классов: {n_unique}")


def evaluate_model(model, data, true_labels, name):
    start_time = time.time()
    model.fit(data)
    exec_time = time.time() - start_time

    pred_labels = model.labels_

    ari = metrics.adjusted_rand_score(true_labels, pred_labels)
    ami = metrics.adjusted_mutual_info_score(true_labels, pred_labels)

    print(f"\nРезультаты для {name}:")
    print(f"  Время работы: {exec_time:.5f} сек")
    print(f"  ARI (Adjusted Rand Index): {ari:.4f}")
    print(f"  AMI (Adjusted Mutual Info): {ami:.4f}")
    return model, ari, ami, exec_time

print("\n2. Обучение KMeans (init='k-means++')...")
kmeans_pp = KMeans(init='k-means++', n_clusters=n_unique, n_init=10, random_state=42)
res_pp = evaluate_model(kmeans_pp, data, labels, "KMeans (k-means++)")

print("\n3. Определение оптимального числа кластеров (Локоть и Силуэт)...")
inertia = []
silhouette_scores = []
k_range = range(2, 16)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(data)
    inertia.append(km.inertia_)
    silhouette_scores.append(metrics.silhouette_score(data, km.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(k_range, inertia, 'bo-')
ax1.set_xlabel('Число кластеров (k)')
ax1.set_ylabel('Inertia (Сумма квадратов расстояний)')
ax1.set_title('Метод локтя')
ax1.grid(True)

ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Число кластеров (k)')
ax2.set_ylabel('Коэффициент силуэта')
ax2.set_title('Метод силуэта')
ax2.grid(True)

plt.tight_layout()
plt.show()
print("Графики построены. На графике локтя видно перегиб, а силуэт максимален при оптимальном k.")

print("\n4. Обучение KMeans (init='random')...")
kmeans_rand = KMeans(init='random', n_clusters=n_unique, n_init=10, random_state=42)
res_rand = evaluate_model(kmeans_rand, data, labels, "KMeans (random)")

print("\n5. Применение PCA и инициализация центров компонентами...")
pca = PCA(n_components=n_unique)
pca.fit(data)

centers_pca = pca.components_

kmeans_pca = KMeans(init=centers_pca, n_clusters=n_unique, n_init=1, random_state=42)
res_pca = evaluate_model(kmeans_pca, data, labels, "KMeans (PCA components)")

print("\nСравнение результатов:")
print(f"{'Метод':<20} | {'ARI':<10} | {'AMI':<10} | {'Время (с)':<10}")
print("-" * 55)
print(f"{'k-means++':<20} | {res_pp[1]:.4f}     | {res_pp[2]:.4f}     | {res_pp[3]:.5f}")
print(f"{'random':<20} | {res_rand[1]:.4f}     | {res_rand[2]:.4f}     | {res_rand[3]:.5f}")
print(f"{'PCA components':<20} | {res_pca[1]:.4f}     | {res_pca[2]:.4f}     | {res_pca[3]:.5f}")

print("\n6. Визуализация кластеров на 2D плоскости...")

pca_2d = PCA(n_components=2)
reduced_data = pca_2d.fit_transform(data)

kmeans_vis = KMeans(init='k-means++', n_clusters=n_unique, n_init=10, random_state=42)
kmeans_vis.fit(reduced_data)

h = .02  
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired, aspect='auto', origin='lower')

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, s=10, cmap='tab10', edgecolors='k', alpha=0.7)

centroids = kmeans_vis.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='white', zorder=10,
            label='Centroids')

plt.title('Визуализация кластеров цифр (PCA 2D)\n(Цвета областей — предсказанные кластеры, точки — истинные классы)')
plt.legend()
plt.show()


print("\n" + "=" * 60)
print("ЧАСТЬ 2. Индивидуальное задание: Датасет CMC (Contraceptive Method Choice)")
print("=" * 60)

column_names = [
    'Wife_age', 'Wife_education', 'Husband_education', 'Num_children_born',
    'Wife_religion', 'Wife_working', 'Husband_occupation', 'Standard_of_living_index',
    'Media_exposure', 'Contraceptive_method_used'
]

try:
    df = pd.read_csv('cmc.data', names=column_names, header=None)
    print(f"Данные успешно загружены из файла 'cmc.data'. Размер: {df.shape}")
except FileNotFoundError:
    print("Файл 'cmc.data' не найден.")
    print("Пожалуйста, убедитесь, что файл лежит в папке с программой.")
    df = pd.DataFrame()

if not df.empty:
    print("\nРаспределение классов (Contraceptive_method_used):")
    print(df['Contraceptive_method_used'].value_counts())
    X = df.drop(columns=['Contraceptive_method_used'])
    y_true = df['Contraceptive_method_used']
    X_scaled = scale(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_

    print(f"\nРезультаты PCA:")
    print(f"  Главная компонента 1 объясняет: {explained_variance[0] * 100:.2f}% дисперсии")
    print(f"  Главная компонента 2 объясняет: {explained_variance[1] * 100:.2f}% дисперсии")
    print(f"  Суммарно 2D объясняют: {sum(explained_variance) * 100:.2f}% дисперсии")
    print(f"  Собственные числа: {eigenvalues}")

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.5)
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')
    plt.title('PCA: Датасет CMC (цветом обозначены истинные классы)')
    plt.legend(*scatter.legend_elements(), title="Классы")
    plt.grid(True)
    plt.show()

    print("\nКластеризация CMC...")

    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, 'bx-')
    plt.xlabel('Число кластеров (k)')
    plt.ylabel('Inertia')
    plt.title('Метод локтя для CMC датасета')
    plt.show()

    n_clusters = 3
    print(f"Выбираем количество кластеров: {n_clusters} (основываясь на данных описания).")

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    ari = metrics.adjusted_rand_score(y_true, clusters)
    ami = metrics.adjusted_mutual_info_score(y_true, clusters)
    silhouette = metrics.silhouette_score(X_scaled, clusters)

    print(f"\nМетрики качества кластеризации:")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Adjusted Mutual Info (AMI): {ami:.4f}")
    print(f"  Silhouette Score: {silhouette:.4f}")

    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='rainbow', alpha=0.6)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=200, c='black', marker='X', label='Centroids')
    plt.title(f'KMeans кластеризация CMC (k={n_clusters})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()
