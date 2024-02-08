import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Класс реализующий метод роя частиц (Particle Swarm Optimization - PSO)
class ParticleSwarmOptimization:
    def __init__(self, num_particles, num_dimensions, num_iterations, min_range, max_range,
                 inertia_weight, cognitive_weight, social_weight):
        # Создание частиц в рое и инициализация параметров
        self.particles = [Particle(num_dimensions, min_range, max_range) for _ in range(num_particles)]
        self.num_iterations = num_iterations
        self.min_range = min_range
        self.max_range = max_range
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    # Метод для выполнения оптимизации
    def optimize(self, objective_function):
        global_best_position = self.particles[0].position.copy()
        global_best_value = float('inf')
        all_positions = []

        # Цикл оптимизации по числу итераций
        for iteration in range(self.num_iterations):
            # Обновление лучшего значения и позиции глобального лучшего решения
            for particle in self.particles:
                particle.new_personal_best(objective_function)
                if particle.personal_best_value < global_best_value:
                    global_best_value = particle.personal_best_value
                    global_best_position = particle.personal_best_position.copy()

            # Обновление позиций частиц и запись их в массив all_positions
            for particle in self.particles:
                particle.new_position(global_best_position, self.min_range, self.max_range,
                                         self.inertia_weight, self.cognitive_weight, self.social_weight)
                all_positions.append(particle.position.copy())

        return global_best_position, global_best_value, all_positions

# Класс представляющий отдельную частицу в PSO
class Particle:
    def __init__(self, num_dimensions, min_range, max_range):
        # Инициализация позиции, скорости и лучшего значения частицы
        self.position = np.random.uniform(low=min_range, high=max_range, size=num_dimensions)
        self.velocity = np.random.rand(num_dimensions)
        self.personal_best_position = self.position.copy()
        self.personal_best_value = float('inf')

    # Метод для обновления скорости частицы
    def new_velocity(self, global_best_position, inertia_weight, cognitive_weight, social_weight):
        inertia_term = inertia_weight * self.velocity
        cognitive_term = cognitive_weight * np.random.rand() * (self.personal_best_position - self.position)
        social_term = social_weight * np.random.rand() * (global_best_position - self.position)
        new_velocity = inertia_term + cognitive_term + social_term
        return new_velocity

    # Метод для обновления позиции частицы
    def new_position(self, global_best_position, min_range, max_range, inertia_weight, cognitive_weight, social_weight):
        self.velocity = self.new_velocity(global_best_position, inertia_weight, cognitive_weight, social_weight)
        self.position += self.velocity
        self.position = np.clip(self.position, min_range, max_range)

    # Метод для обновления лучшего значения частицы
    def new_personal_best(self, objective_function):
        current_value = objective_function(self.position)
        if current_value < self.personal_best_value:
            self.personal_best_value = current_value
            self.personal_best_position = self.position.copy()

# Класс представляющий графический интерфейс для PSO
class PSO_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Particle Swarm Optimization")

        # Создание элементов управления для параметров PSO
        self.selected_function = ttk.Combobox(root, values=['f(x, y) = -12*x2 +4*x1^2 +4*x2^2-4*x1*x2'])
        self.inertia_weight_entry = ttk.Entry(root)
        self.cognitive_weight_entry = ttk.Entry(root)
        self.social_weight_entry = ttk.Entry(root)
        self.num_particles_entry = ttk.Entry(root)
        self.num_iterations_entry = ttk.Entry(root)

        # Установка значений по умолчанию
        self.selected_function.set('f(x, y) = -12*x2 +4*x1^2 +4*x2^2-4*x1*x2')
        self.inertia_weight_entry.insert(0, '0.5')
        self.cognitive_weight_entry.insert(0, '1.5')
        self.social_weight_entry.insert(0, '1.5')
        self.num_particles_entry.insert(0, '30')
        self.num_iterations_entry.insert(0, '100')

        # Создание кнопки для запуска оптимизации
        calculate_button = ttk.Button(root, text="Вычислить", command=self.perform_optimization)

        # Размещение элементов управления на форме с использованием сетки
        ttk.Label(root, text="Функция:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.selected_function.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(root, text="Коэффициент инерции:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.inertia_weight_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(root, text="Коэффициент собственного лучшего значения:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.cognitive_weight_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(root, text="Коэффициент глобального лучшего значения:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.social_weight_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(root, text="Количество частиц:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.num_particles_entry.grid(row=4, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(root, text="Количество итераций:").grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.num_iterations_entry.grid(row=5, column=1, padx=5, pady=5, sticky='w')
        calculate_button.grid(row=6, column=0, columnspan=2, pady=10)

        # Поля для вывода результатов
        self.result_label = ttk.Label(root, text="")
        self.result_label.grid(row=7, column=0, columnspan=2, pady=10)

        # Место для графика
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=2, rowspan=8, padx=10, pady=10)

    # Функция для выполнения оптимизации при нажатии кнопки
    def perform_optimization(self):
        try:
            # Получение параметров из элементов управления
            selected_function = self.selected_function.get()
            inertia_weight = float(self.inertia_weight_entry.get())
            cognitive_weight = float(self.cognitive_weight_entry.get())
            social_weight = float(self.social_weight_entry.get())
            num_particles = int(self.num_particles_entry.get())
            num_iterations = int(self.num_iterations_entry.get())

            # Определение функции для оптимизации (в данном случае только одна функция)
            if selected_function == 'f(x, y) = -12*x2 +4*x1^2 +4*x2^2-4*x1*x2':
                objective_function = lambda x: -12*(x[1]) +4*(x[0])**2 +4*(x[1])**2-4*x[0]*x[1]


            # Создание экземпляра PSO и выполнение оптимизации
            pso = ParticleSwarmOptimization(num_particles, 2, num_iterations, -100, 100,
                                            inertia_weight, cognitive_weight, social_weight)
            best_solution, best_value, all_positions = pso.optimize(objective_function)

            # Вывод результатов
            result_text = f"Лучшее решение: {best_solution}\nЛучшее значение: {best_value}"
            self.result_label.config(text=result_text)

            # Построение графика
            self.ax.clear()
            self.ax.set_title('График решений')
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            all_positions = np.array(all_positions)

            for i in range(len(all_positions)):
                self.ax.scatter(all_positions[i, 0], all_positions[i, 1], color='green', alpha=0.5)

            self.ax.scatter(best_solution[0], best_solution[1], label='Лучшее решение', color='red')
            self.ax.legend()

            self.canvas.draw()

        except Exception as e:
            self.result_label.config(text=f"Ошибка: {str(e)}")


# Основная часть программы
if __name__ == "__main__":
    root = tk.Tk()
    app = PSO_GUI(root)
    root.mainloop()