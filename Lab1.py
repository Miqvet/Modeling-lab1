import math
import matplotlib.pyplot as plt


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


def main():
    with open('data.txt', 'r') as f:
        data = list(map(float, f.read().replace(',', '.').split()))

    task_characteristics(data, [10, 20, 50, 100, 200, 300])
    # task_values_plot(data)
    # task_autocorrelation_analysis(data)
    # task_frequency_distribution_histogram(data)
    # task_distribution_law_approximation(data)
    #
    # data1 = task_law_generate_random(300)
    # task_characteristics(data1, [10, 20, 50, 100, 200, 300])
    # task_autocorrelation_analysis(data1)
    #
    # task_comparative_analysis(data, data1)
    # task_correlation_dependence(data, data1)


if __name__ == '__main__':
    main()

'''
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
'''
