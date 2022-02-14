from typing import Optional
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

class ActorCriticController:
    def __init__(self, environment, learning_rate: float, discount_factor: float) -> None:
        self.environment = environment
        self.discount_factor: float = discount_factor
        self.model: tf.keras.Model = self.create_actor_critic_model()
        self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # stosuję optymalizator Adam tak, jak jest to opisane w zadaniu
        self.log_action_probability: Optional[tf.Tensor] = None  # zmienna pomocnicza, przyda się do obliczania docelowej straty
        self.tape: Optional[tf.GradientTape] = None  # zmienna pomocnicza, związana z działaniem frameworku
        self.last_error_squared: float = 0.0  # zmienna używana do wizualizacji wyników

    @staticmethod
    def create_actor_critic_model() -> tf.keras.Model:
        # Implementuję sieć neuronową zgodnie ze schematem (wersja bez rozłączenia)
        # input_layer = tf.keras.layers.Input(4)  # stan w zadaniu z kijkiem ma 4 elementy
        # first_layer = tf.keras.layers.Dense(1024, activation="relu")(input_layer)
        # first_layer_normalization = tf.keras.layers.LayerNormalization()(first_layer)
        # second_layer = tf.keras.layers.Dense(256, activation='relu')(first_layer_normalization)
        # second_layer_normalization = tf.keras.layers.LayerNormalization()(second_layer)
        # actor = tf.keras.layers.Dense(2, activation="softmax")(second_layer_normalization)  # w zadaniu z kijkiem aktor ma dwa stany
        # critic = tf.keras.layers.Dense(1, activation="linear")(second_layer_normalization)
        # return tf.keras.Model(inputs=input_layer, outputs=(actor, critic))

        # Implementuję sieć neuronową zgodnie ze schematem (wersja z rozłączeniem)
        # input_layer = tf.keras.layers.Input(4)  # stan w zadaniu z kijkiem ma 4 elementy
        # actor_first_layer = tf.keras.layers.Dense(1024, activation="relu")(input_layer)
        # critic_first_layer = tf.keras.layers.Dense(1024, activation="relu")(input_layer)
        # actor_first_layer_normalization = tf.keras.layers.LayerNormalization()(actor_first_layer)
        # critic_first_layer_normalization = tf.keras.layers.LayerNormalization()(critic_first_layer)
        # actor_second_layer = tf.keras.layers.Dense(256, activation='relu')(actor_first_layer_normalization)
        # critic_second_layer = tf.keras.layers.Dense(256, activation='relu')(critic_first_layer_normalization)
        # actor_second_layer_normalization = tf.keras.layers.LayerNormalization()( actor_second_layer)
        # critic_second_layer_normalization = tf.keras.layers.LayerNormalization()(critic_second_layer)
        # actor = tf.keras.layers.Dense(2, activation="softmax")(actor_second_layer_normalization)  # w zadaniu z kijkiem aktor ma dwa stany
        # critic = tf.keras.layers.Dense(1, activation="linear")(critic_second_layer_normalization)
        # return tf.keras.Model(inputs=input_layer, outputs=(actor, critic))

        # Implementuję sieć neuronową zgodnie ze schematem (zmiejszone liczby H1 i H2)
        input_layer = tf.keras.layers.Input(4)  # stan w zadaniu z kijkiem ma 4 elementy
        actor_first_layer = tf.keras.layers.Dense(128, activation="relu")(input_layer)
        critic_first_layer = tf.keras.layers.Dense(128, activation="relu")(input_layer)
        actor_first_layer_normalization = tf.keras.layers.LayerNormalization()(actor_first_layer)
        critic_first_layer_normalization = tf.keras.layers.LayerNormalization()(critic_first_layer)
        actor_second_layer = tf.keras.layers.Dense(32, activation='relu')(actor_first_layer_normalization)
        critic_second_layer = tf.keras.layers.Dense(32, activation='relu')(critic_first_layer_normalization)
        actor_second_layer_normalization = tf.keras.layers.LayerNormalization()( actor_second_layer)
        critic_second_layer_normalization = tf.keras.layers.LayerNormalization()(critic_second_layer)
        actor = tf.keras.layers.Dense(2, activation="softmax")(actor_second_layer_normalization)  # w zadaniu z kijkiem aktor ma dwa stany
        critic = tf.keras.layers.Dense(1, activation="linear")(critic_second_layer_normalization)
        return tf.keras.Model(inputs=input_layer, outputs=(actor, critic))

        # Implementuję sieć neuronową dla problemu lądowania zgodnie ze schematem
        # input_layer = tf.keras.layers.Input(8)  # stan w zadaniu z lądowaniem ma 8 elementów
        # actor_first_layer = tf.keras.layers.Dense(128, activation="relu")(input_layer)
        # critic_first_layer = tf.keras.layers.Dense(128, activation="relu")(input_layer)
        # actor_first_layer_normalization = tf.keras.layers.LayerNormalization()(actor_first_layer)
        # critic_first_layer_normalization = tf.keras.layers.LayerNormalization()(critic_first_layer)
        # actor_second_layer = tf.keras.layers.Dense(32, activation='relu')(actor_first_layer_normalization)
        # critic_second_layer = tf.keras.layers.Dense(32, activation='relu')(critic_first_layer_normalization)
        # actor_second_layer_normalization = tf.keras.layers.LayerNormalization()( actor_second_layer)
        # critic_second_layer_normalization = tf.keras.layers.LayerNormalization()(critic_second_layer)
        # actor = tf.keras.layers.Dense(2, activation="softmax")(actor_second_layer_normalization)  # w zadaniu z kijkiem aktor ma dwa stany
        # critic = tf.keras.layers.Dense(1, activation="linear")(critic_second_layer_normalization)
        # return tf.keras.Model(inputs=input_layer, outputs=(actor, critic))

    def choose_action(self, state: np.ndarray) -> int:
        state = self.format_state(state)  # przygotowanie stanu do formatu akceptowanego przez framework

        self.tape = tf.GradientTape()
        with self.tape:
            # wszystko co dzieje się w kontekście danej taśmy jest zapisywane i może posłużyć do późniejszego wyliczania pożądanych gradientów
            # stosuję klasę pomocniczą tfp.distributions.Categorical zaproponowaną w instrukcji. Po analizie dokumentacji tensorflow wyszukuję funkcję experimental_sample_and_log_prob, która zdaje się spełniać warunki w zadaniu
            temp = tfp.distributions.Categorical(probs=self.model(state)[0][0])
            temp_func = temp.experimental_sample_and_log_prob()
            action = temp_func[0]
            self.log_action_probability = temp_func[1]
        return int(action)

    # noinspection PyTypeChecker
    def learn(self, state: np.ndarray, reward: float, new_state: np.ndarray, terminal: bool) -> None:
        state = self.format_state(state)
        new_state = self.format_state(new_state)

        with self.tape:  # to ta sama taśma, które użyliśmy już w fazie wybierania akcji
            # wszystko co dzieje się w kontekście danej taśmy jest zapisywane i może posłużyć do późniejszego wyliczania pożądanych gradientów
            # if S' nie jest stanem teminalnym then delta <- R + gamma * v^(S', w) = v^(S, w) else delta <- R - v^(S,w) end if
            if not terminal:
                error = reward + self.discount_factor * float(self.model(new_state)[1][0].numpy()) - self.model(state)[1][0]
            else:
                error = reward - self.model(state)[1][0]
            self.last_error_squared = float(error) ** 2
            # L_critic <- delta ** 2
            critic_loss = error ** 2
            # L_actor <- - delta * ln pi^(A|S, w)
            actor_loss = (-1 * float(error.numpy()) * self.log_action_probability)
            loss = critic_loss + actor_loss  # suma strat aktora i krytyka

        gradients = self.tape.gradient(loss, self.model.trainable_weights)  # tu obliczamy gradienty po wagach z naszej straty, pomagają w tym informacje zapisane na taśmie
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))  # tutaj zmieniamy wagi modelu wykonując krok po gradiencie w kierunku minimalizacji straty

    @staticmethod
    def format_state(state: np.ndarray) -> np.ndarray:
        return np.reshape(state, (1, state.size))


def main() -> None:
    environment = gym.make('CartPole-v1')  # zamień na gym.make('LunarLander-v2') by zająć się lądownikiem
    # environment = gym.make('LunarLander-v2')  # zmiana środowiska dla problemu lądowania
    # controller = ActorCriticController(environment, 0.00001, 0.99)
    controller = ActorCriticController(environment, 0.01, 0.99)  # Zmniejszam krok uczący do 0.01
    # controller = ActorCriticController(environment, 0.0000001, 0.99)  # zmniejszam stukrotnie krok uczący dla problemu lądowania

    past_rewards = []
    past_errors = []
    for i_episode in tqdm(range(2000)):  # tu decydujemy o liczbie epizodów
    # for i_episode in tqdm(range(3000)):  # zwiększam liczbę do 3000 w problemie lądowania
        done = False
        state = environment.reset()
        reward_sum = 0.0
        errors_history = []

        while not done:
           #  environment.render()  # tą linijkę możemy wykomentować, jeżeli nie chcemy mieć wizualizacji na żywo

            action = controller.choose_action(state)
            new_state, reward, done, info = environment.step(action)
            controller.learn(state, reward, new_state, done)
            state = new_state
            reward_sum += reward
            errors_history.append(controller.last_error_squared)

        past_rewards.append(reward_sum)
        past_errors.append(np.mean(errors_history))

        window_size = 50  # tutaj o rozmiarze okienka od średniej kroczącej
        if i_episode % 25 == 0:  # tutaj o częstotliwości zrzucania wykresów
            if len(past_rewards) >= window_size:
                fig, axs = plt.subplots(2)
                axs[0].plot(
                    [np.mean(past_errors[i:i + window_size]) for i in range(len(past_errors) - window_size)],
                    'tab:red',
                )
                axs[0].set_title('mean squared error')
                axs[1].plot(
                    [np.mean(past_rewards[i:i + window_size]) for i in range(len(past_rewards) - window_size)],
                    'tab:green',
                )
                axs[1].set_title('sum of rewards')
            plt.savefig(f'plots/learning_{i_episode}.png')
            plt.clf()

    environment.close()
    controller.model.save("final.model")  # tu zapisujemy model


if __name__ == '__main__':
    main()
