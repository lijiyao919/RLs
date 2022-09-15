from per import PERAgent
import gym

env_name = "CartPole-v0"
env = gym.make(env_name)
env.seed(0)

train_step = 50000
log_feq = 1000

def train():
    agent = PERAgent(env.observation_space.shape[0], env.action_space.n, 256, 0.001, batch_size=32, target_update_feq=1000, eps_end=0.05, eps_decay=10000)
    agent.train_mode()
    i_step = 0
    episode = 0

    while i_step < train_step:
        state = env.reset()
        done = False
        episode_reward = 0
        episode += 1
        while not done:
            action = agent.select_action(state, i_step)
            next_state, reward, done, _ = env.step(action)
            episode_reward+=reward
            #env.render()

            agent.store_exp(state, action, reward, next_state, int(done))
            agent.learn(i_step, log_feq)
            state = next_state
            i_step += 1
        print(episode, episode_reward)


if __name__ == '__main__':
    train()