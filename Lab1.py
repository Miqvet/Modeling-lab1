import math
import matplotlib.pyplot as plt
import random


def mean(data):
    return sum(data) / len(data)


def variance(data, mean_value):
    return sum((x - mean_value) ** 2 for x in data) / len(data)


def std_deviation(variance_value):
    return math.sqrt(variance_value)


def coefficient_of_variation(std_dev, mean_value):
    return std_dev / mean_value


def confidence_interval(mean_value, std_dev, n, alpha):
    student_value = {
        0.9: [1.833, 1.729, 1.6766, 1.6604, 1.6525, 1.65],
        0.95: [2.262, 2.093, 2.0096, 1.984, 1.972, 1.968],
        0.99: [3.25, 2.861, 2.68, 2.626, 2.601, 2.592]
    }[alpha][[10, 20, 50, 100, 200, 300].index(n)]

    margin_of_error = student_value * (std_dev / math.sqrt(n))
    return mean_value - margin_of_error, mean_value + margin_of_error, margin_of_error


def task_characteristics(_data: list, sizes: list):
    results = []

    for n in sizes:
        data = _data[:n]

        mean_value = mean(data)
        variance_value = variance(data, mean_value)
        std_dev = std_deviation(variance_value)
        cv = coefficient_of_variation(std_dev, mean_value)
        confidence_intervals = {alpha: confidence_interval(mean_value, std_dev, n, alpha) for alpha in
                                [0.9, 0.95, 0.99]}

        results.append({'n': n,
                        'mean': mean_value,
                        'variance': variance_value,
                        'std_dev': std_dev,
                        'cv': cv})

        print(f"Выборка размера {n}:")
        print(f"  Математическое ожидание: {mean_value:.4f}")
        print(f"  Дисперсия: {variance_value:.4f}")
        print(f"  Среднеквадратическое отклонение: {std_dev:.4f}")
        print(f"  Коэффициент вариации: {cv:.4f}")
        for alpha, interval in confidence_intervals.items():
            print(f"  Доверительный интервал для ɑ={alpha}: ({interval[0]:.4f}, {interval[1]:.4f}) (±{interval[2]:.4f} от мат. ожидания)")
            results[-1][f'confidence_interval_{alpha}_margin'] = interval[2]
        print()

    reference_result = results[-1]
    for result in results[:-1]:
        print(f"Относительные отклонения характеристик выборки из {result['n']} элементов от выборки в {reference_result['n']} элементов:")
        del result['n']
        for key in result.keys():
            reference_value, value = reference_result[key], result[key]
            deviation = abs((value - reference_value) / reference_value) * 100
            print(f"  {key}: {deviation:.4f}%")
        print()


def task_values_plot(data):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(data) + 1), data, label="Числовая последовательность", color='blue')

    # Настройка графика
    plt.title("График числовой последовательности")
    plt.xlabel("Индекс элемента")
    plt.ylabel("Значение элемента")
    plt.grid(True)
    plt.legend()
    plt.show()


# Функция для вычисления автокорреляции
def autocorrelation(data, lag):
    """
    Вычисление автокорреляции для заданного лага.
    """
    n = len(data)
    mean_value = sum(data) / n
    numerator = sum((data[i] - mean_value) * (data[i + lag] - mean_value) for i in range(n - lag))
    denominator = sum((x - mean_value) ** 2 for x in data)
    
    return numerator / denominator


# Функция для выполнения автокорреляционного анализа
def task_autocorrelation_analysis(data, max_lag=20):
    """
    Выполнение автокорреляционного анализа до указанного лага.
    Строим график автокорреляции для различных лагов.
    """
    autocorrelations = [autocorrelation(data, lag) for lag in range(1, max_lag + 1)]

    # Построение графика автокорреляции
    plt.figure(figsize=(10, 6))
    plt.stem(range(1, max_lag + 1), autocorrelations, use_line_collection=True)
    plt.title("График автокорреляции")
    plt.xlabel("Лаг")
    plt.ylabel("Автокорреляция")
    plt.grid(True)
    plt.show()
    return autocorrelations


def task_frequency_distribution_histogram(data):
    """
    Построение гистограммы распределения частот.
    """
    N = len(data)
    bins = 1 + int(math.log2(N))
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, density=True, color='blue', alpha=0.7) 
    plt.title("Гистограмма распределения вероятностей")
    plt.xlabel("Значения")
    plt.ylabel("Вероятность")
    plt.grid(True)
    plt.show()
    

# Функции для распределений

def uniform_distribution(min_value, max_value, size):
    """Генерация равномерного распределения"""
    return [random.uniform(min_value, max_value) for _ in range(size)]

def exponential_distribution(lambda_value, size):
    """Генерация экспоненциального распределения"""
    return [-math.log(1 - random.random()) / lambda_value for _ in range(size)]

def erlang_distribution(k, lambda_value, size):
    """Генерация распределения Эрланга k-го порядка"""
    return [sum([-math.log(1 - random.random()) / lambda_value for _ in range(k)]) for _ in range(size)]

def hyperexponential_distribution(p, lambda_1, lambda_2, size):
    """Генерация гиперэкспоненциального распределения"""
    return [(random.choice([lambda_1, lambda_2]) * -math.log(1 - random.random())) for _ in range(size)]


# Аппроксимация распределения по коэффициенту вариации

def approximate_distribution(data):
    """Выбор и генерация аппроксимации распределения в зависимости от коэффициента вариации"""
    mean_value = mean(data)
    variance_value = variance(data, mean_value)
    std_dev = std_deviation(variance_value)
    cv = coefficient_of_variation(std_dev, mean_value)
    
    size = len(data)
    
    # 1. CV ≈ 0
    if cv < 0.1:
        min_value = min(data)
        max_value = max(data)
        approx_data = uniform_distribution(min_value, max_value, size)
        title = "Равномерное распределение"
    
    # 2. CV ≈ 1
    elif abs(cv - 1) < 0.1:
        lambda_value = 1 / mean_value
        approx_data = exponential_distribution(lambda_value, size)
        title = "Экспоненциальное распределение"
    
    # 3. CV < 1
    elif cv < 1:
        k = round(1 / cv ** 2)
        lambda_value = k / mean_value
        approx_data = erlang_distribution(k, lambda_value, size)
        title = f"Эрланговское распределение k={k}"
    
    # 4. CV > 1
    else:
        lambda_1 = 2 / mean_value
        lambda_2 = lambda_1 / (cv ** 2 - 1)
        p = 0.5  # Вероятность выбора одного из параметров
        approx_data = hyperexponential_distribution(p, lambda_1, lambda_2, size)
        title = "Гиперэкспоненциальное распределение"
    
    return approx_data, title

# Построение гистограммы и аппроксимации

def task_distribution_law_approximation(data):
    """
    Построение гистограммы исходных данных и гистограммы аппроксимированного распределения.
    """
    # Количество элементов
    N = len(data)
    
    # Количество бинов по правилу Стёрджеса
    bins = 1 + int(math.log2(N))
    
    # Аппроксимация данных
    approx_data, title = approximate_distribution(data)
    
    # Построение гистограммы исходных данных
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, density=True, color='blue', alpha=0.6, label="Исходные данные")
    
    # Построение гистограммы аппроксимации
    plt.hist(approx_data, bins=bins, density=True, color='red', alpha=0.4, label=f"Аппроксимировано: {title}")
    
    # Оформление графика
    plt.title("Гистограмма распределения и его аппроксимация")
    plt.xlabel("Значения")
    plt.ylabel("Плотность вероятности")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    with open('O:\\Itmo\\5_SEM\\Modeling\\Modeling-lab1\\data.txt', 'r') as f:
        data = list(map(float, f.read().replace(',', '.').split()))

    task_characteristics(data, [10, 20, 50, 100, 200, 300])
    task_values_plot(data)
    task_autocorrelation_analysis(data)
    task_frequency_distribution_histogram(data)
    task_distribution_law_approximation(data) # не уверен стоит проверить
    #
    # data1 = task_law_generate_random(300)
    # task_characteristics(data1, [300])
    # task_autocorrelation_analysis(data1)
    #
    # task_comparative_analysis(data, data1)
    # task_correlation_dependence(data, data1)


if __name__ == '__main__':
    main()












