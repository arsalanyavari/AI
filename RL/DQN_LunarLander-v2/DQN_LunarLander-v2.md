## Solve LunarLander-v2 game using DQN.


```python
# For Debian based (needed packages)
#! sudo apt-get install build-essential libc6-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev libjpeg-dev libpng-dev

! pip install gym
! pip install pygame
! pip install gym[box2d]
```


```python
import gymnasium as gym
from stable_baselines3 import DQN
import os
```


```python
train_env = gym.make("LunarLander-v2")
test_env = gym.make("LunarLander-v2", render_mode="human")
```


```python
save_path = "./dqn_lunar_lander_model"
model_exist = os.path.exists(f"{save_path}.zip")
```


```python
if model_exist:
    print("Loading existing model...")
    model = DQN.load(save_path, env=train_env)
else:
    print("Creating a new model...")
    model = DQN(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=0.001,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        target_update_interval=250
    )

    timesteps_to_train = 2000000
```

    Loading existing model...
    Wrapping the env with a `Monitor` wrapper
    Wrapping the env in a DummyVecEnv.


### Train the model


```python
if not model_exist:
    print(f"Training the model for {timesteps_to_train} timesteps...")
    model.learn(total_timesteps=timesteps_to_train)
    model.save(save_path)
```

### Test the model performance


```python
episodes = 10
total_rewards = []

print("\nStarting test episodes...\n")

for episode in range(episodes):
    obs, _ = test_env.reset()
    episode_reward = 0
    terminated = truncated = False
    step = 0

    print(f"\nStarting Episode {episode + 1}\n")

    while not (terminated or truncated):
        step += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action) 
        episode_reward += reward

        test_env.render()

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1} finished with Reward = {episode_reward}")

test_env.close()

```

    
    Starting test episodes...
    
    
    Starting Episode 1
    
    Episode 1 finished with Reward = 282.5193047112648
    
    Starting Episode 2
    
    Episode 2 finished with Reward = 278.58507224927223
    
    Starting Episode 3
    
    Episode 3 finished with Reward = 231.4056407939228
    
    Starting Episode 4
    
    Episode 4 finished with Reward = 307.4485807746782
    
    Starting Episode 5
    
    Episode 5 finished with Reward = 262.6625937808301
    
    Starting Episode 6
    
    Episode 6 finished with Reward = 257.99842571722263
    
    Starting Episode 7
    
    Episode 7 finished with Reward = 217.92687647541175
    
    Starting Episode 8
    
    Episode 8 finished with Reward = 281.46275594814426
    
    Starting Episode 9
    
    Episode 9 finished with Reward = 271.4860365967917
    
    Starting Episode 10
    
    Episode 10 finished with Reward = 278.6101341362273



```python
average_reward = sum(total_rewards) / episodes
print(f"\nAverage Reward over {episodes} episodes: {average_reward}")

# Set the success threshold
if average_reward > 200:
    print("Test successful! The agent performs well.")
else:
    print("Test incomplete. More training needed.")
```

    
    Average Reward over 10 episodes: 267.0105421183766
    Test successful! The agent performs well.


## If you want to play it yourself, run the block below :D


```python
import gymnasium as gym
from gymnasium.utils.play import play
play(gym.make("LunarLander-v2", render_mode="rgb_array"), keys_to_action={"w": 2, "a": 1, "d": 3}, noop=0)
```


```python

```
