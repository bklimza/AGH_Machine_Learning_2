from __future__ import annotations
import collections
import random

import numpy as np
import sklearn.preprocessing as skl_preprocessing

from problem import Action, available_actions, Corner, Driver, Experiment, Environment, State

ALMOST_INFINITE_STEP = 100000
MAX_LEARNING_STEPS = 500


class RandomDriver(Driver):
    def __init__(self):
        self.current_step: int = 0

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        return random.choice(available_actions(state))

    def control(self, state: State, last_reward: int) -> Action:
        self.current_step += 1
        return random.choice(available_actions(state))

    def finished_learning(self) -> bool:
        return self.current_step > MAX_LEARNING_STEPS


class OffPolicyNStepSarsaDriver(Driver):
    def __init__(self, step_size: float, step_no: int, experiment_rate: float, discount_factor: float) -> None:
        self.step_size: float = step_size
        self.step_no: int = step_no
        self.experiment_rate: float = experiment_rate
        self.discount_factor: float = discount_factor
        self.q: dict[tuple[State, Action], float] = collections.defaultdict(float)
        self.current_step: int = 0
        self.final_step: int = ALMOST_INFINITE_STEP
        self.finished: bool = False
        self.states: dict[int, State] = dict()
        self.actions: dict[int, Action] = dict()
        self.rewards: dict[int, int] = dict()

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        self.states[self._access_index(self.current_step)] = state
        action = self._select_action(self.epsilon_greedy_policy(state, available_actions(state)))
        self.actions[self._access_index(self.current_step)] = action
        self.final_step = ALMOST_INFINITE_STEP
        self.finished = False
        return action

    def control(self, state: State, last_reward: int) -> Action:
        if self.current_step < self.final_step:
            self.rewards[self._access_index(self.current_step + 1)] = last_reward
            self.states[self._access_index(self.current_step + 1)] = state
            if self.final_step == ALMOST_INFINITE_STEP and (
                    last_reward == 0 or self.current_step == MAX_LEARNING_STEPS
            ):
                self.final_step = self.current_step
            action = self._select_action(self.epsilon_greedy_policy(state, available_actions(state)))
            self.actions[self._access_index(self.current_step + 1)] = action
        else:
            action = Action(0, 0)

        update_step = self.current_step - self.step_no + 1
        if update_step >= 0:
            return_value_weight = self._return_value_weight(update_step)
            return_value = self._return_value(update_step)
            state_t = self.states[self._access_index(update_step)]
            action_t = self.actions[self._access_index(update_step)]
            # if t + n < T, then G <- G + y ** n * Q(S_(t+n), A_(t+n))
            if update_step + self.step_no < self.final_step:
                return_value += (self.discount_factor ** self.step_no) * self.q[self.states[self._access_index(update_step + self.step_no)], self.actions[self._access_index(update_step + self.step_no)]]
            # Q(S_t, A_t) <- Q(S_t, A_t) + a * p * [G - Q(S_t, A_t)]
            self.q[state_t, action_t] = self.q[state_t, action_t] + self.step_size * return_value_weight * (return_value - self.q[state_t, action_t])

        if update_step == self.final_step - 1:
            self.finished = True

        self.current_step += 1
        return action

    def _return_value(self, update_step):
        return_value = 0.0
        # G <- SUM[i = r + 1][min(t + n, T] y ** (i - r - 1) * R_i
        for i in range(update_step + 1, min(update_step + self.step_no, self.final_step)):
            return_value += (self.discount_factor ** (i - update_step - 1)) * self.rewards[self._access_index(i)]
        return return_value

    def _return_value_weight(self, update_step):
        return_value_weight = 1.0
        # p <- PI[i = r + 1][min(t + n -1, T -1)] pi(A_i|S_i) / b(A_i|S_i)
        for i in range(update_step + 1, min(update_step + self.step_no - 1, self.final_step - 1)):
            S_t, A_t = self.states[self._access_index(i)], self.actions[self._access_index(i)]
            b, pi = self.epsilon_greedy_policy(S_t, available_actions(S_t))[A_t], self.greedy_policy(S_t, available_actions(S_t))[A_t]
            return_value_weight = return_value_weight * (pi / b)
        return return_value_weight

    def finished_learning(self) -> bool:
        return self.finished

    def _access_index(self, index: int) -> int:
        return index % (self.step_no + 1)

    @staticmethod
    def _select_action(actions_distribution: dict[Action, float]) -> Action:
        actions = list(actions_distribution.keys())
        probabilities = list(actions_distribution.values())
        i = np.random.choice(list(range(len(actions))), p=probabilities)
        return actions[i]

    def epsilon_greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        # # Usually we can designate epsilon greedy policy in this way:
        # # If uniform random number between 0 and 1 < epsilon, then random action. Else best action

        # if random.uniform(0, 1) < self.experiment_rate:
        #     probabilities = self._random_probabilities(actions)
        # else:
        #     probabilities = self._greedy_probabilities(state, actions)

        # I tried this method and I had problem with "ValueError: probabilities do not sum to 1"
        # Probability summation works fine:
        probabilities = (1.0 - self.experiment_rate) * self._greedy_probabilities(state, actions) + self.experiment_rate * self._random_probabilities(actions)
        return {action: probability for action, probability in zip(actions, probabilities)}

    def greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        probabilities = self._greedy_probabilities(state, actions)
        return {action: probability for action, probability in zip(actions, probabilities)}

    def _greedy_probabilities(self, state: State, actions: list[Action]) -> np.ndarray:
        values = [self.q[state, action] for action in actions]
        maximal_spots = (values == np.max(values)).astype(float)
        return self._normalise(maximal_spots)

    @staticmethod
    def _random_probabilities(actions: list[Action]) -> np.ndarray:
        maximal_spots = np.array([1.0 for _ in actions])
        return OffPolicyNStepSarsaDriver._normalise(maximal_spots)

    @staticmethod
    def _normalise(probabilities: np.ndarray) -> np.ndarray:
        return skl_preprocessing.normalize(probabilities.reshape(1, -1), norm='l1')[0]


def main() -> None:
    experiment = Experiment(
        environment=Environment(
            corner=Corner(
                name='corner_d'
            ),
            steering_fail_chance=0.01,
        ),
        driver=OffPolicyNStepSarsaDriver(
            step_no=4,
            step_size=0.26,
            experiment_rate=0.05,
            discount_factor=1.00,
        ),
        number_of_episodes=10000,
    )

    experiment.run()


if __name__ == '__main__':
    main()
