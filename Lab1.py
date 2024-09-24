import math
import matplotlib.pyplot as plt
import random
import numpy as np

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
        confidence_intervals = {alpha: confidence_interval(mean_value, std_dev, n, alpha)
                                for alpha in [0.9, 0.95, 0.99]}

        results.append({'n': n,
                        'mean': mean_value,
                        'variance': variance_value,
                        'std_dev': std_dev,
                        'cv': cv})

        print(f"\nВыборка размера {n}:")
        print(f"  Математическое ожидание: {mean_value:.4f}")
        print(f"  Дисперсия: {variance_value:.4f}")
        print(f"  Среднеквадратическое отклонение: {std_dev:.4f}")
        print(f"  Коэффициент вариации: {cv:.4f}")
        for alpha, interval in confidence_intervals.items():
            print(f"  Доверительный интервал для ɑ={alpha}: ({interval[0]:.4f}, {interval[1]:.4f}) (±{interval[2]:.4f} от мат. ожидания)")
            results[-1][f'confidence_interval_{alpha}_margin'] = interval[2]

    reference_result = results[-1]
    for result in results[:-1]:
        print(f"\nОтносительные отклонения характеристик выборки из {result['n']} элементов от выборки в {reference_result['n']} элементов:")
        del result['n']
        for key in result.keys():
            reference_value, value = reference_result[key], result[key]
            deviation = abs((value - reference_value) / reference_value) * 100
            print(f"  {key}: {deviation:.4f}%")


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


def task_autocorrelation_analysis(data):
    n = len(data)
    mean_value = sum(data) / n
    var_value = sum((x - mean_value) ** 2 for x in data)

    lags = 10
    autocorr_values = [sum((data[i] - mean_value) * (data[i + lag] - mean_value) for i in range(n - lag)) / var_value
                       for lag in range(1, lags + 1)]

    print("\nЗначения автокорреляции:")
    for i, autocorr in enumerate(autocorr_values, 1):
        print(f"Лаг {i}: {autocorr:.4f}")

    plt.figure(figsize=(10, 5))
    plt.stem(range(1, lags + 1), autocorr_values)
    plt.xticks(range(1, lags + 1))
    plt.title("Автокорреляционная функция (ACF)")
    plt.xlabel("Лаг")
    plt.ylabel("Коэффициент автокорреляции")
    plt.grid(True)
    plt.show()

    threshold = 2 / (n ** 0.5)  # Порог для случайной автокорреляции
    if all(abs(autocorr) < threshold for autocorr in autocorr_values):
        print("Последовательность можно считать случайной.")
    else:
        print("Последовательность содержит зависимости, не является случайной.")


def task_frequency_distribution_histogram(data):
    n = len(data)
    bins = 1 + int(math.log2(n))

    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, density=True, color='blue', alpha=0.7)
    plt.title("Гистограмма распределения вероятностей")
    plt.xlabel("Значения")
    plt.ylabel("Вероятность")
    plt.grid(True)
    plt.show()


# Функции для вычисления статистических показателей
def uniform_pdf(x, min_value, max_value):
    if min_value <= x <= max_value:
        return 1 / (max_value - min_value)
    else:
        return 0

def exponential_pdf(x, lambda_value):
    if x >= 0:
        return lambda_value * math.exp(-lambda_value * x)
    else:
        return 0

def erlang_pdf(x, k, lambda_value):
    if x >= 0:
        return (lambda_value ** k * x ** (k - 1) * math.exp(-lambda_value * x)) / math.factorial(k - 1)
    else:
        return 0

def hyperexponential_pdf(x, p, lambda_1, lambda_2):
    if x >= 0:
        return p * lambda_1 * math.exp(-lambda_1 * x) + (1 - p) * lambda_2 * math.exp(-lambda_2 * x)
    else:
        return 0


# Построение гистограммы и аппроксимирующей кривой
def plot_histogram_and_approximation(data):
    N = len(data)
    bins = 1 + int(math.log2(N))

    mean_value = mean(data)
    variance_value = variance(data, mean_value)
    std_dev = std_deviation(variance_value)
    cv = coefficient_of_variation(std_dev, mean_value)

    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, density=True, color='blue', alpha=0.7, label="Исходные данные")
    
    x_values = np.linspace(min(data), max(data), 1000)
    y_values = []
    
    if cv < 0.1:
        title = "Равномерное распределение"
        min_value, max_value = min(data), max(data)
        y_values = [uniform_pdf(x, min_value, max_value) for x in x_values]
    
    elif abs(cv - 1) < 0.1:
        title = "Экспоненциальное распределение"
        lambda_value = 1 / mean(data)
        y_values = [exponential_pdf(x, lambda_value) for x in x_values]
    
    elif cv < 1:
        title = f"Эрланговское распределение k={round(1 / cv ** 2)}"
        k = round(1 / coefficient_of_variation(std_deviation(variance(data, mean(data))), mean(data)) ** 2)
        lambda_value = k / mean(data)
        y_values = [erlang_pdf(x, k, lambda_value) for x in x_values]
    
    else:
        title = "Гиперэкспоненциальное распределение"
        lambda_1 = 2 / mean(data)
        lambda_2 = lambda_1 / (coefficient_of_variation(std_deviation(variance(data, mean(data))), mean(data)) ** 2 - 1)
        p = 0.5  # Вероятность выбора одного из параметров
        y_values = [hyperexponential_pdf(x, p, lambda_1, lambda_2) for x in x_values]
    
    plt.plot(x_values, y_values, color='red', label=f"Аппроксимирующая кривая: {title}")
    
    plt.title("Гистограмма распределения и аппроксимирующая кривая")
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
    plot_histogram_and_approximation(data) #объединение пред пункта и посл пункта

    # data1 = task_law_generate_random(300)
    # task_characteristics(data1, [10, 20, 50, 100, 200, 300])
    # task_autocorrelation_analysis(data1)

    # task_comparative_analysis(data, data1)
    # task_correlation_dependence(data, data1)


if __name__ == '__main__':
    main()
