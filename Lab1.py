import math
import matplotlib.pyplot as plt

def mean(data):
    """Математическое ожидание"""
    return sum(data) / len(data)

def variance(data, mean_value):
    """Дисперсия"""
    return sum((x - mean_value) ** 2 for x in data) / len(data)

def std_deviation(variance_value):
    """Среднеквадратическое отклонение"""
    return math.sqrt(variance_value)

def coefficient_of_variation(std_dev, mean_value):
    """Коэффициент вариации"""
    return std_dev / mean_value

def confidence_interval(mean_value, std_dev, n, z_value):
    """Доверительный интервал"""
    margin_of_error = z_value * (std_dev / math.sqrt(n))
    return mean_value - margin_of_error, mean_value + margin_of_error

def relative_deviation(value, reference_value):
    """Относительное отклонение"""
    return abs((value - reference_value) / reference_value) * 100

# Коэфициенты Стьюдента для 10 20 50 100 200 300
z_values = {
    0.9: [1.833, 1.729, 1.6766, 1.6604, 1.6525, 1.65],
    0.95: [2.262, 2.093, 2.0096, 1.984, 1.972, 1.968],
    0.99: [3.25, 2.861, 2.68, 2.626, 2.601, 2,592]
}

# Чтение данных из файла
def read_data_from_file(file_path):
    """Чтение данных из файла, предположим, что данные разделены пробелами"""
    with open(file_path, 'r') as f:
        return list(map(float, f.read().replace(',', '.').split()))


def analyze_data(data):
    """Основной анализ данных"""
    n = len(data)
    mean_value = mean(data)
    variance_value = variance(data, mean_value)
    std_dev = std_deviation(variance_value)
    cv = coefficient_of_variation(std_dev, mean_value)
    confidence_intervals = {alpha: confidence_interval(mean_value, std_dev, n, z[[10, 20, 50, 100, 200, 300].index(n)])
                            for alpha, z in z_values.items()}

    return {
        'mean': mean_value,
        'variance': variance_value,
        'std_dev': std_dev,
        'cv': cv,
        'confidence_intervals': confidence_intervals
    }

# Функция для анализа частичных выборок
def analyze_partial_data(data, sizes):
    """Анализ данных для подвыборок размера sizes"""
    results = {}
    for size in sizes:
        subset = data[:size]
        results[size] = analyze_data(subset)
    return results

# Чтение основного файла и анализ
data = read_data_from_file('data.txt')

# Анализ данных для выборок из 10, 100 и всей выборки
results = analyze_partial_data(data, [10, 20, 50, 100, 200, 300])

# Вывод результатов
for size, res in results.items():
    print(f"Выборка размера {size}:")
    print(f"  Математическое ожидание: {res['mean']}")
    print(f"  Дисперсия: {res['variance']}")
    print(f"  Среднеквадратическое отклонение: {res['std_dev']}")
    print(f"  Коэффициент вариации: {res['cv']}")
    for alpha, interval in res['confidence_intervals'].items():
        print(f"  Доверительный интервал для {alpha*100}%: {interval}")
    print()

# Относительные отклонения по сравнению с полной выборкой
reference = results[300]
for size in [10, 20, 50, 100, 200]:
    print(f"Относительные отклонения для выборки из {size} элементов:")
    for key in ['mean', 'variance', 'std_dev', 'cv']:
        deviation = relative_deviation(results[size][key], reference[key])
        print(f"  {key}: {deviation:.2f}%")


def plot_sequence(data):
    """
    Построение графика для числовой последовательности.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(data) + 1), data, label="Числовая последовательность", color='blue')

    # Настройка графика
    plt.title("График числовой последовательности")
    plt.xlabel("Индекс элемента")
    plt.ylabel("Значение элемента")
    plt.grid(True)
    plt.legend()

    # Отображение графика
    plt.show()

plot_sequence(data)