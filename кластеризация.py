# Импорт ннеобходимых модулей
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Определение параметров
CLUSTER_NUM: int = 6  # Кол-во кластеров
CLUSTER_POWER: int = 200  # Кол-во точек в кластере
DISPERSION: float = 0.3  # Разброс точек в кластере
ITERATIONS: int = 10  # Кол-во итераций
CLUSTER_ORDINATE: list[int] = [np.random.randint(0, 10) * 2 for _ in range(CLUSTER_NUM)]
COLORS: list[str] = ["b", "g", "r", "c", "m", "y", "k", "w"][:CLUSTER_NUM]
THRESHOLD: float = CLUSTER_POWER * DISPERSION * CLUSTER_NUM * 1.5
print("Порог:", THRESHOLD, "WCSS")

# Создание кластеров с помощью распределения Гаусса
cluster_points: np.ndarray = np.zeros(shape=(CLUSTER_NUM * CLUSTER_POWER, 2))
for cluster in range(CLUSTER_NUM):
    for point in range(CLUSTER_POWER):
        cluster_points[cluster * CLUSTER_POWER + point][0] = np.random.normal(
            cluster, DISPERSION, (1,)
        )
        cluster_points[cluster * CLUSTER_POWER + point][1] = np.random.normal(
            CLUSTER_ORDINATE[cluster], DISPERSION, (1,)
        )
cluster_points = cluster_points.T

# Определение диапазона k-точек
K_RANGE: list[float] = [
    np.max(cluster_points[0]) - np.min(cluster_points[0]),
    np.max(cluster_points[1]) - np.min(cluster_points[1]),
]


# Функция, вычисляющая новые k-точки
def find_clusters(k_points: np.ndarray) -> tuple[list, np.ndarray, np.ndarray]:
    global CLUSTER_NUM
    clusters: list = [
        [] for _ in range(CLUSTER_NUM)
    ]  # Пустые кластеры, определяемые k-алгоритмом

    for undefined_point in cluster_points.T:
        # Вектор расстояний от каждой точки до k-точек
        distances: np.ndarray = np.empty((CLUSTER_NUM,))
        for k_point_index in range(CLUSTER_NUM):
            # Поиск расстояния с помощью нормы вектора
            distances[k_point_index] = np.linalg.norm(
                undefined_point - k_points.T[k_point_index], 2
            )

        # Причисление к определенному кластеру точки в зависимости от найденного расстояния
        clusters[np.argmin(distances)].append(undefined_point)

    new_k_points: np.ndarray = np.zeros(shape=(CLUSTER_NUM, 2))

    # Вычисление качества k-точек
    quality: np.ndarray = np.zeros((CLUSTER_NUM,))
    for cluster_num in range(CLUSTER_NUM):
        for cluster_point in clusters[cluster_num]:
            quality[cluster_num] += np.linalg.norm(
                k_points.T[cluster_num] - cluster_point, 2
            )

    # Изменение существующих k-точек на средние точки каждого соответствующего кластера
    for k_point_index in range(CLUSTER_NUM):
        cluster: np.ndarray = np.array(clusters[k_point_index])

        if len(cluster) == 0:  # Обработка пустого кластера
            new_k_points[k_point_index] = k_points.T[k_point_index]
        else:
            # Изменение k-точки на новую
            new_k_points[k_point_index] = cluster.mean(axis=0)

    return clusters, new_k_points.T, quality


attempts: int = 1  # Номер попытки
while True:  # Вечный цикл для поиска наилучших k-точек
    # Массивы данных для построения графика
    animate_scatter_data: list = []
    animate_k_data: list = []
    quality_plot_data: list = []

    # Определение случайных k-точек
    k_points: np.ndarray = np.array(
        [
            np.random.random((CLUSTER_NUM)) * (K_RANGE[0]) + np.min(cluster_points[0]),
            np.random.random((CLUSTER_NUM)) * (K_RANGE[1]) + np.min(cluster_points[1]),
        ]
    )
    for i in range(ITERATIONS):
        # Вывод информации в консоль
        print(" " * 100, end="\r")
        print(
            f"ATTEMPT: {attempts}\tITERSTION: {i+1}/{ITERATIONS}\tQUALITY: {0 if len(quality_plot_data) == 0 else quality_plot_data[-1]}",
            end="\r",
        )

        animate_k_data.append(k_points)

        # Вызов функции поиска новых k-точек
        clusters: list
        quality: np.ndarray
        clusters, k_points, quality = find_clusters(k_points)
        quality_plot_data.append(np.sum(quality))
        animate_scatter_data.append(clusters)

    # Проверка качества новых точек (середин кластеров)
    if quality_plot_data[-1] <= THRESHOLD:
        break

    attempts += 1
print()

# Определение поля для постоения графиков
fig, ax = plt.subplots(ncols=2, nrows=1)
y_limits: list = [min(quality_plot_data) - 20, max(quality_plot_data) + 20]


# Функция обновления анимации
def update(frame: int):
    global ax, ITERATIONS
    ax[0].cla()
    ax[1].cla()

    # Определение диапазонов и названий осей первого графика
    ax[0].set_xlim(0, ITERATIONS)
    ax[0].set_ylim(y_limits[0], y_limits[1])
    ax[0].set_xlabel("Итерация")
    ax[0].set_ylabel("WCSS")

    # Определение названий осей второго графика
    ax[0].set_title("График ошибки системы")
    ax[1].set_title("Кластеры")

    # Построение графиков на основе данных из
    # списков animate_scatter_data, animate_k_data и quality_plot_data

    ax[0].plot(quality_plot_data[:frame])
    for plot_data, color_ind in list(
        zip(animate_scatter_data[frame], range(len(COLORS)))
    ):
        plot_data = np.array(plot_data)
        try:
            ax[1].scatter(*plot_data.T, c=COLORS[color_ind])
        except:
            pass

        ax[1].scatter(
            *animate_k_data[frame][:, color_ind],
            s=150,
            c=COLORS[color_ind],
            edgecolors="black",
        )


ax[0].plot(quality_plot_data)
for plot_data, color_ind in list(zip(clusters, range(len(COLORS)))):
    plot_data: np.ndarray = np.array(plot_data)
    try:
        ax[1].scatter(*plot_data.T, c=COLORS[color_ind])
    except:
        pass

    ax[1].scatter(
        *k_points[:, color_ind],
        s=150,
        c=COLORS[color_ind],
        edgecolors="black",
    )

# Вывод графика/создание и сохранение анмации
animation = FuncAnimation(fig, update, frames=ITERATIONS, interval=100)
animation.save("anim2.gif")
