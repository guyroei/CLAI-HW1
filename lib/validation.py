import numpy as np

import torch

def validation_run(env, net, Actions, episodes=100, device="cpu", epsilon=0.02, commission=0.1):
    stats = {
        'episode_reward': [],
        'episode_steps': [],
        'order_profits': [],
        'order_steps': [],
    }

    for episode in range(episodes):
        obs, _ = env.reset()

        total_reward = 0.0
        position = None
        position_steps = None
        episode_steps = 0

        while True:
            obs_v = torch.tensor(np.array([obs])).to(device)
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = Actions(action_idx)

            close_price = env._state._cur_close()

            if action == Actions.Buy and position is None:
                position = close_price
                position_steps = 0
            elif action == Actions.Close and position is not None:
                profit = close_price - position - (close_price + position) * commission / 100
                profit = 100.0 * profit / position
                stats['order_profits'].append(profit)
                stats['order_steps'].append(position_steps)
                position = None
                position_steps = None

            obs, reward, done, truncated, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            if position_steps is not None:
                position_steps += 1
            if done:
                if position is not None:
                    profit = close_price - position - (close_price + position) * commission / 100
                    profit = 100.0 * profit / position
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)
                break

        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    return {key: np.mean(vals) for key, vals in stats.items()}

def validation_run_2(env, net, ExtendedActions, episodes=100, device="cpu", epsilon=0.02, commission=0.1):
    stats = {
        'episode_reward': [],
        'episode_steps': [],
        'order_profits': [],
        'order_steps': [],
    }
    extendedActionsInstance = ExtendedActions()
    for episode in range(episodes):
        obs, _ = env.reset()

        total_reward = 0.0
        position = 0.0
        position_steps = 0
        episode_steps = 0

        while True:
            obs_v = torch.tensor(np.array([obs])).to(device)
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = extendedActionsInstance.actionIdxToAction(action_idx)

            close_price = env._state._cur_close()
            money_pool = env._state.moneyPool
            owned_shares = env._state.ownShares

            num_of_shares_to_buy = action[0]
            num_of_shares_to_sell = action[1]
            if num_of_shares_to_buy > 0:
                position = close_price * num_of_shares_to_buy
                if position > money_pool:
                    position = 0.0
                position_steps = 0
            if num_of_shares_to_sell > 0 and num_of_shares_to_sell <= owned_shares and position > 0.0:
                profit = (close_price * num_of_shares_to_sell) - position - ((close_price * num_of_shares_to_sell) + position) * commission / 100
                profit = 100.0 * profit / position
                stats['order_profits'].append(profit)
                stats['order_steps'].append(position_steps)
                position = 0.0
                position_steps = 0

            obs, reward, done, truncated, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            if position_steps > 0:
                position_steps += 1
            if done:
                if position > 0.0:
                    profit = (close_price * num_of_shares_to_sell) - position - ((close_price * num_of_shares_to_sell) + position) * commission / 100
                    profit = 100.0 * profit / position
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)
                break

        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    return {key: np.mean(vals) for key, vals in stats.items()}