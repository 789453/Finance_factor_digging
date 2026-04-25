import torch
from alpha_gfn.env.core import GFNEnvCore
from alphagen.data.expression import Feature, Operators
from alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManager
from gfn.samplers import Sampler

def debug_env():
    # Mock components
    class MockPool:
        def try_new_expr(self, expr, **kwargs):
            print(f"Evaluating: {expr}")
            return 0.1, 0.1
            
    device = torch.device("cpu")
    features = ["open", "close", "high", "low", "volume"]
    operators = [Operators.Add, Operators.Sub, Operators.Mul, Operators.Div, Operators.Delta, Operators.TsSum]
    delta_times = [10, 20]
    constants = [1.0, 2.0]
    
    env = GFNEnvCore(
        pool=MockPool(),
        device=device,
        custom_features=features,
        operators=operators,
        delta_times=delta_times,
        constants=constants,
        max_expr_length=10
    )
    
    sampler = DiscreteActionsSampler()
    
    print(f"Action space size: {env.n_actions}")
    print(f"Tokens: {env._tokens}")
    
    # Test step-by-step
    states = env.reset(batch_shape=(1,))
    print(f"Initial state: {states.tensor}")
    
    for i in range(15):
        mask = env.get_mask(states)
        print(f"Step {i}, Mask (first 10): {mask[0, :10]}")
        
        # Pick a valid action
        valid_actions = mask[0].nonzero().flatten()
        if len(valid_actions) == 0:
            print("No valid actions!")
            break
        
        action = valid_actions[0] # Just pick first valid
        print(f"Selected action {action}: {env._tokens[action]}")
        
        states = env.step(states, action.unsqueeze(0))
        print(f"New state: {states.tensor}")
        
        if states.is_terminal.any():
            print("Terminal reached!")
            expr = env.state_to_expression(states.tensor[0])
            print(f"Final Expression: {expr}")
            reward = env.reward(states)
            print(f"Reward: {reward}")
            break

if __name__ == "__main__":
    debug_env()
