from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "member2_ai"))

from routing_environment import RoutingEnvironment
from dqn_agent import DQNAgent


def main():
    env = RoutingEnvironment(seed=7)
    agent = DQNAgent(env.state_size, env.action_size, batch_size=4)
    state = env.reset(source="R1", destination="R6")

    assert len(state) == env.state_size, f"Expected {env.state_size} state features, got {len(state)}"
    valid_actions = env.valid_actions()
    assert valid_actions, "Environment returned no valid neighboring routers"

    action = agent.choose_action(state, valid_actions, training=False)
    assert action in valid_actions, "Agent selected a router that is not a neighbor"

    next_state, reward, done, info = env.step(action)
    assert len(next_state) == env.state_size, "Next-state size changed unexpectedly"
    assert isinstance(reward, float), "Reward should be numeric"
    assert isinstance(done, bool), "Done flag should be boolean"
    assert isinstance(info, dict), "Step info should be a dictionary"

    print("Integration interface test PASSED.")
    print("Routers:", env.ROUTERS)
    print("State size:", env.state_size)
    print("Current router: R1")
    print("Valid next-hop options:", env.path_names(valid_actions))
    print("DQN selected:", env.id_to_router[action])
    print("Reward returned:", round(reward, 2))
    print("The shared environment and DQN interface are connected.")
    print("Note: this smoke test does not measure learning quality.")


if __name__ == "__main__":
    main()
