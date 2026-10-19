# RL Workbench

An interactive reinforcement learning laboratory for learning how an agent turns experience into a policy. Select an environment, inspect the method, train a tabular agent, then evaluate and compare the results.

The project is structured as a course mini-project: it connects the problem definition, RL formulation, implementation, experiments, and evidence that can be discussed in a technical report or viva.

## What you can explore

- **Seven classic methods:** Value Iteration, Policy Iteration, Monte Carlo, TD(0), n-step TD, SARSA, and Q-Learning.
- **Twenty-six environment configurations:** custom GridWorlds, mazes, navigation tasks, TicTacToe, and Gymnasium benchmarks.
- **Interactive concept lab:** follow the state → action → reward → update loop and adjust values in a live Bellman update sandbox.
- **Training and evaluation:** inspect reward and episode-length curves, evaluate a learned policy, and generate training and inference videos.
- **Experiment analysis:** browse saved runs, compare algorithms, and review the effect of parameter changes.
- **Dark animated interface:** motion respects the device's reduced-motion setting.

## Project formulation

### Mini-project problem

In a navigation task, an agent must reach a goal while learning which actions produce useful outcomes. The agent observes a state, chooses an action, receives a reward, and updates its value estimates. The workbench makes these decisions visible and lets a learner compare planning and model-free methods.

### Example: GridWorld

| Element | Formulation |
| --- | --- |
| State | The agent's discrete grid position |
| Actions | Up, right, down, left |
| Reward | +1 at the goal and −0.01 per non-terminal step |
| Episode end | The agent reaches the goal |
| Baselines to compare | SARSA (on-policy) and Q-Learning (off-policy) |
| Evidence | Episode reward, episode length, policy/value plots, and evaluation episodes |

The environment and reward details above describe the custom GridWorld implementation. Other environments have their own transition and reward rules. Record the selected environment and parameters with every reported result.

## Learning path

1. **Learn:** choose a method and connect its update equation to the current environment.
2. **Configure:** select an environment, algorithm, and hyperparameters in the left panel.
3. **Train:** collect episode-level metrics and inspect the learned values or policy.
4. **Evaluate:** run the learned policy and review its behavior and performance.
5. **Analyze:** compare runs and discuss the effect of method and parameter choices.

For a fair comparison, use the same environment, training budget, and evaluation conditions. Repeat stochastic runs when reporting results and describe the variation; one run alone is not evidence of general performance.

## Run locally

Python 3.11 is recommended for the pinned dependencies.

```bash
git clone https://github.com/Shreyansh123185655/RL-Workbench.git
cd RL-Workbench
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

The repository includes the Streamlit entry point, dependency list, and dark theme configuration needed by Community Cloud.

1. Push the project to a GitHub repository you administer.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and choose **Create app**.
3. Select this repository, the branch to deploy, and `app.py` as the entry point.
4. In **Advanced settings**, choose Python **3.11**, then deploy.

Community Cloud needs access to the source repository; the account connected to Streamlit must have permission to deploy it. See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Course report outline

Use the course's current submission requirements. A mini-project report can document:

1. **Title and abstract:** problem, objective, method, and measured outcome.
2. **Introduction:** motivation, problem definition, and objectives.
3. **Methodology:** state/action spaces, reward function, algorithm, hyperparameters, training strategy, and software.
4. **Results:** training curves, policy behavior, evaluation results, baseline comparison, and interpretation.
5. **Conclusion:** limitations and future scope.
6. **References:** cite papers, packages, datasets, code, and figures used.

The supplied course brief specifies a maximum of four pages for a mini-project report (references excluded), and four to six pages for a research or review paper. Follow faculty instructions and use actual results from your own runs; the workbench does not claim experimental results on your behalf.

## Repository map

```text
app.py                         Streamlit entry point
interface/page_config.py      Theme, layout, and shared styles
interface/sidebar.py          Environment, method, and parameter controls
interface/tabs/learning_tab.py RL concepts and update sandbox
interface/tabs/training_tab.py Training controls and result views
interface/tabs/inference_tab.py Policy evaluation and video generation
interface/tabs/design_tab.py  Training history and comparisons
rl/core/                      Dynamic programming and tabular RL methods
rl/envs/                      Custom and Gymnasium environments
requirements.txt              Python dependencies
```

## Scope and limitations

- The included learning methods use tabular value or action-value estimates; this is an educational workbench, not a deep RL framework.
- Continuous observations are discretized by the Gymnasium wrapper, which can limit performance and scale poorly as the state dimension grows.
- Training history and generated media are stored by the running app. Ephemeral hosting filesystems may not preserve them across restarts.
- Results depend on the random seed, environment configuration, and hyperparameters. Report those settings with any comparison.

## References

- Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018. [Online edition](http://incompleteideas.net/book/the-book-2nd.html).
- Farama Foundation, [Gymnasium documentation](https://gymnasium.farama.org/).
- Streamlit, [Community Cloud documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud).
