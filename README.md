# pgpe_light

A watered down verison of the excelent [nnaisense/pgpelib](https://github.com/nnaisense/pgpelib) that I use in small projects.


```python
from pgpelib.pgpe import PGPE


solver = PGPE(
        solution_length=number_of_parameters,
        popsize=my_population_size,
        optimizer="clipup",
        optimizer_config=dict(max_speed=0.25, momentum=0.9),
        center_learning_rate=0.15,
        stdev_init=0.05,
        dtype=np.float32,
    )
```

A rough code skeleton would look like this:

```python
class Agent:
    def __init__(self, number_of_parameters):
        self.weights = np.random.randn(number_of_parameters).astype(np.float32)
        self.number_of_parameters = number_of_parameters

    def load_weights(self, weights):
        self.weights = weights

    def evaluate_state(self, state):
        vector_of_meaningful_values = self.get_inputs_from(state)
        return np.dot(self.weights, vector_of_meaningful_values)

    def your_search_algo_goes_here(self, state):
        # minmax, GA, MCTS, etc
        # the evaluation function is the one that uses the weights
        return best_action_for_this_turn


class Engine:
    ...

    def play_game(self, agent_0, agent_1):
        # play a game between two agents
        return agent_0_score, agent_1_score  # could be 1 if won, 0 if lost, or a some score


class Driver:
    def __init__(self, my_population_size, number_of_parameters):
        self.agent = [Agent(number_of_parameters) for _ in range(my_population_size)]
        self.solver = PGPE(
            solution_length=number_of_parameters,
            popsize=my_population_size,
            optimizer="clipup",
            optimizer_config=dict(max_speed=0.25, momentum=0.9),  # max speed of weights change, how much of the previous change to keep
            center_learning_rate=0.15,  # how 'fast' will weights change
            stdev_init=0.05,  # how much noise to start with
            dtype=np.float32,
        )

    def run(self):

        while True:
            weights_to_test = self.solver.ask()

            for i, agent in enumerate(self.agent):
                agent.load_weights(weights_to_test[i])

            list_all_games = []
            for i, agent_0 in enumerate(self.agent):
                for j, agent_1 in enumerate(self.agent):
                    if i != j:
                        list_all_games.append((i, j))

            results = np.zeros(len(self.agent))

            for i, j in list_all_games:
                agent_0_score, agent_1_score = self.engine.play_game(self.agent[i], self.agent[j])
                results[i] += agent_0_score
                results[j] += agent_1_score

            self.solver.tell(results)

            # maybe play some game between an agent with the center (solver.center) as weights and the best previous agent
            # if the center is better, it becomes the new best agent
            # log progress vs previous best, make a pretty graph, etc
            # save weights of best agents to disk
```
