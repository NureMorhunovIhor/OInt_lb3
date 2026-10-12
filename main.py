import os
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from deap import base, creator, tools, algorithms


warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



FULL_MODE = False

if FULL_MODE:
    POPULATION_SIZE = 20
    GENERATIONS = 50
    GA_EPOCHS = 15
else:
    POPULATION_SIZE = 8
    GENERATIONS = 5
    GA_EPOCHS = 10


CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.3



print("=" * 70)
print("1. ЗАВАНТАЖЕННЯ ДАТАСЕТУ")
print("=" * 70)

dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

print("Кількість об'єктів:", X.shape[0])
print("Кількість ознак:", X.shape[1])
print("Кількість класів:", len(np.unique(y)))
print("Класи:", dataset.target_names)




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print("\nРозмір тренувальної вибірки:", X_train.shape)
print("Розмір тестової вибірки:", X_test.shape)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nМасштабування StandardScaler виконано.")



ACTIVATIONS = [
    "relu",
    "tanh",
    "sigmoid"
]

OPTIMIZERS = [
    "adam",
    "sgd",
    "rmsprop"
]

LEARNING_RATES = [
    0.001,
    0.005,
    0.01
]

BATCH_SIZES = [
    16,
    32,
    64
]


def create_model(
        num_layers,
        neurons1,
        neurons2,
        neurons3,
        activation,
        optimizer_name,
        learning_rate
):

    model = Sequential()

    model.add(Input(shape=(X_train.shape[1],)))

    model.add(
        Dense(
            neurons1,
            activation=activation
        )
    )

    if num_layers >= 2:
        model.add(
            Dense(
                neurons2,
                activation=activation
            )
        )

    if num_layers >= 3:
        model.add(
            Dense(
                neurons3,
                activation=activation
            )
        )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    if optimizer_name == "adam":
        optimizer = Adam(
            learning_rate=learning_rate
        )

    elif optimizer_name == "sgd":
        optimizer = SGD(
            learning_rate=learning_rate
        )

    else:
        optimizer = RMSprop(
            learning_rate=learning_rate
        )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model



print("\n")
print("=" * 70)
print("2. НАВЧАННЯ БАЗОВОЇ НЕЙРОННОЇ МЕРЕЖІ")
print("=" * 70)

baseline_model = create_model(
    num_layers=1,
    neurons1=32,
    neurons2=32,
    neurons3=32,
    activation="relu",
    optimizer_name="adam",
    learning_rate=0.001
)

baseline_model.summary()

baseline_model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

baseline_loss, baseline_accuracy = baseline_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nРезультат базової моделі:")
print(f"Loss:     {baseline_loss:.4f}")
print(f"Accuracy: {baseline_accuracy:.4f}")
print(
    f"Точність: {baseline_accuracy * 100:.2f}%"
)

if not hasattr(creator, "FitnessMax"):
    creator.create(
        "FitnessMax",
        base.Fitness,
        weights=(1.0,)
    )

if not hasattr(creator, "Individual"):
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMax
    )

toolbox = base.Toolbox()

def create_individual():

    return creator.Individual([
        random.randint(1, 3),
        random.choice([8, 16, 32, 64, 128]),
        random.choice([8, 16, 32, 64, 128]),
        random.choice([8, 16, 32, 64, 128]),
        random.randint(
            0,
            len(ACTIVATIONS) - 1
        ),

        random.randint(
            0,
            len(OPTIMIZERS) - 1
        ),

        random.randint(
            0,
            len(LEARNING_RATES) - 1
        ),

        random.randint(
            0,
            len(BATCH_SIZES) - 1
        )
    ])


toolbox.register(
    "individual",
    create_individual
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)

evaluation_number = 0

def evaluate_individual(individual):

    global evaluation_number

    evaluation_number += 1

    num_layers = individual[0]

    neurons1 = individual[1]
    neurons2 = individual[2]
    neurons3 = individual[3]

    activation = ACTIVATIONS[
        individual[4]
    ]

    optimizer_name = OPTIMIZERS[
        individual[5]
    ]

    learning_rate = LEARNING_RATES[
        individual[6]
    ]

    batch_size = BATCH_SIZES[
        individual[7]
    ]

    print(
        f"\nОцінювання особини №{evaluation_number}"
    )

    print(
        f"Шари: {num_layers}, "
        f"нейрони: [{neurons1}, "
        f"{neurons2}, "
        f"{neurons3}]"
    )

    print(
        f"Activation: {activation}, "
        f"Optimizer: {optimizer_name}, "
        f"LR: {learning_rate}, "
        f"Batch: {batch_size}"
    )
    tf.keras.backend.clear_session()

    model = create_model(
        num_layers=num_layers,
        neurons1=neurons1,
        neurons2=neurons2,
        neurons3=neurons3,
        activation=activation,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=GA_EPOCHS,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop]
    )

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(
        f"Accuracy = {accuracy:.4f}"
    )

    return (accuracy,)


toolbox.register(
    "evaluate",
    evaluate_individual
)

toolbox.register(
    "mate",
    tools.cxTwoPoint
)


def mutate_individual(
        individual,
        mutation_probability=0.2
):

    if random.random() < mutation_probability:
        individual[0] = random.randint(1, 3)

    if random.random() < mutation_probability:
        individual[1] = random.choice(
            [8, 16, 32, 64, 128]
        )

    if random.random() < mutation_probability:
        individual[2] = random.choice(
            [8, 16, 32, 64, 128]
        )

    if random.random() < mutation_probability:
        individual[3] = random.choice(
            [8, 16, 32, 64, 128]
        )

    if random.random() < mutation_probability:
        individual[4] = random.randint(
            0,
            len(ACTIVATIONS) - 1
        )

    if random.random() < mutation_probability:
        individual[5] = random.randint(
            0,
            len(OPTIMIZERS) - 1
        )

    if random.random() < mutation_probability:
        individual[6] = random.randint(
            0,
            len(LEARNING_RATES) - 1
        )

    if random.random() < mutation_probability:
        individual[7] = random.randint(
            0,
            len(BATCH_SIZES) - 1
        )

    return (individual,)


toolbox.register(
    "mutate",
    mutate_individual
)

toolbox.register(
    "select",
    tools.selTournament,
    tournsize=3
)

print("\n")
print("=" * 70)
print("3. ЗАПУСК ГЕНЕТИЧНОГО АЛГОРИТМУ")
print("=" * 70)

print(
    f"\nРозмір популяції: {POPULATION_SIZE}"
)

print(
    f"Кількість поколінь: {GENERATIONS}"
)

population = toolbox.population(
    n=POPULATION_SIZE
)

stats = tools.Statistics(
    lambda individual:
    individual.fitness.values[0]
)

stats.register(
    "avg",
    np.mean
)

stats.register(
    "std",
    np.std
)

stats.register(
    "min",
    np.min
)

stats.register(
    "max",
    np.max
)


hall_of_fame = tools.HallOfFame(1)

population, logbook = algorithms.eaSimple(
    population,
    toolbox,

    cxpb=CROSSOVER_PROBABILITY,

    mutpb=MUTATION_PROBABILITY,

    ngen=GENERATIONS,

    stats=stats,

    halloffame=hall_of_fame,

    verbose=True
)

best_individual = hall_of_fame[0]

best_layers = best_individual[0]

best_neurons1 = best_individual[1]
best_neurons2 = best_individual[2]
best_neurons3 = best_individual[3]

best_activation = ACTIVATIONS[
    best_individual[4]
]

best_optimizer = OPTIMIZERS[
    best_individual[5]
]

best_learning_rate = LEARNING_RATES[
    best_individual[6]
]

best_batch_size = BATCH_SIZES[
    best_individual[7]
]


print("\n")
print("=" * 70)
print("4. НАЙКРАЩІ ГІПЕРПАРАМЕТРИ")
print("=" * 70)

print(
    "Найкраща хромосома:",
    list(best_individual)
)

print(
    f"Кількість прихованих шарів: "
    f"{best_layers}"
)

print(
    f"Нейрони першого шару: "
    f"{best_neurons1}"
)

if best_layers >= 2:
    print(
        f"Нейрони другого шару: "
        f"{best_neurons2}"
    )

if best_layers >= 3:
    print(
        f"Нейрони третього шару: "
        f"{best_neurons3}"
    )

print(
    f"Функція активації: "
    f"{best_activation}"
)

print(
    f"Оптимізатор: "
    f"{best_optimizer}"
)

print(
    f"Learning rate: "
    f"{best_learning_rate}"
)

print(
    f"Batch size: "
    f"{best_batch_size}"
)

print(
    f"Fitness: "
    f"{best_individual.fitness.values[0]:.4f}"
)


print("\n")
print("=" * 70)
print("5. НАВЧАННЯ КІНЦЕВОЇ МОДЕЛІ")
print("=" * 70)

tf.keras.backend.clear_session()

final_model = create_model(
    num_layers=best_layers,
    neurons1=best_neurons1,
    neurons2=best_neurons2,
    neurons3=best_neurons3,
    activation=best_activation,
    optimizer_name=best_optimizer,
    learning_rate=best_learning_rate
)

final_model.summary()

history = final_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=best_batch_size,
    validation_split=0.2,
    verbose=0
)

final_loss, final_accuracy = final_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n")
print("=" * 70)
print("6. ПОРІВНЯННЯ РЕЗУЛЬТАТІВ")
print("=" * 70)

print(
    f"Базова модель:       "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"Оптимізована модель: "
    f"{final_accuracy * 100:.2f}%"
)

difference = (
    final_accuracy -
    baseline_accuracy
) * 100

print(
    f"Зміна точності: "
    f"{difference:+.2f}%"
)

if difference > 0:
    print(
        "Генетичний алгоритм покращив "
        "точність нейронної мережі."
    )

elif difference == 0:
    print(
        "Точність моделей однакова."
    )

else:
    print(
        "У цьому запуску базова модель "
        "показала вищу точність."
    )


generations = logbook.select("gen")

average_fitness = logbook.select("avg")
max_fitness = logbook.select("max")

plt.figure(figsize=(10, 6))

plt.plot(
    generations,
    average_fitness,
    marker="o",
    label="Середня точність"
)

plt.plot(
    generations,
    max_fitness,
    marker="s",
    label="Найкраща точність"
)

plt.xlabel("Покоління")
plt.ylabel("Accuracy")

plt.title(
    "Зміна точності в процесі "
    "генетичної оптимізації"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "genetic_algorithm_evolution.png",
    dpi=300
)

plt.show()


plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Навчальна вибірка"
)

plt.plot(
    history.history["val_accuracy"],
    label="Валідаційна вибірка"
)

plt.xlabel("Епоха")
plt.ylabel("Accuracy")

plt.title(
    "Точність кінцевої нейронної мережі"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "final_model_accuracy.png",
    dpi=300
)

plt.show()

print("\n")
print("=" * 70)
print("РОБОТУ ЗАВЕРШЕНО")
print("=" * 70)

print(
    f"""
Базова точність:
{baseline_accuracy * 100:.2f}%

Оптимізована точність:
{final_accuracy * 100:.2f}%

Різниця:
{difference:+.2f}%

Найкращі параметри:

Кількість шарів:
{best_layers}

Нейрони:
{best_neurons1},
{best_neurons2},
{best_neurons3}

Функція активації:
{best_activation}

Оптимізатор:
{best_optimizer}

Learning rate:
{best_learning_rate}

Batch size:
{best_batch_size}
"""
)