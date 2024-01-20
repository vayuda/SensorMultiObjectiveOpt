from tqdm import tqdm
import json
import torch


def test(env, user, agent, n_episodes, episode_len):
    a1tr = 0
    a1teff = 0
    test_data = env.generate_user_trajectories(user, n_episodes, episode_len)
    for episode in (test_data):
        r1, eff1 = env.sim_episode(episode, agent, False)
        a1tr += r1
        a1teff +=(eff1/len(episode))
    return a1tr, a1teff

def train(cfg):
    results = []
    for cfg_value in cfg['var. param vals']:
        cfg['var. param name'] = cfg_value

        env = cfg['env']
        user = cfg['user']
        agent1 = cfg['agent1']
        agent2 = cfg['agent2']

        a1train_rs = []
        a2train_rs = []
        
        a1test_rs = []
        a2test_rs = []

        a1test_eff = []
        a2test_eff = []

        user_trajectories = env.generate_user_trajectories(user, cfg['# train ep'], cfg['ep len'])
        for n, episode in enumerate(tqdm(user_trajectories)):

            a1r, _ = env.sim_episode(episode, agent1, True)
            a2r , _ = env.sim_episode(episode, agent2, True)

            a1train_rs.append(a1r)
            a2train_rs.append(a2r)

            if not n % cfg['test every']:
                a1tr, a1teff = test(env, user, agent1, cfg['# test ep'], cfg['ep len'])
                a2tr, a2teff = test(env, user, agent2, cfg['# test ep'], cfg['ep len'])
                a1test_rs.append(a1tr/cfg['# test ep'])
                a1test_eff.append(a1teff/cfg['# test ep'])
                a2test_rs.append(a2tr/cfg['# test ep'])
                a2test_eff.append(a2teff/cfg['# test ep'])



        if cfg['save path']:
            if agent1.cfg['savable net']:
                torch.save(agent1.net.state_dict(), cfg['save path'] + cfg['exp name'] + 'final.pt')
            
            if agent2.cfg['savable net']:
                torch.save(agent2.net.state_dict(), cfg['save path'] + cfg['exp name'] + 'final.pt')

        a1tr, a1teff = test(env, user, agent1, cfg['# test ep'], cfg['ep len'])
        a2tr, a2teff = test(env, user, agent2, cfg['# test ep'], cfg['ep len'])
        a1test_rs.append(a1tr/cfg['# test ep'])
        a1test_eff.append(a1teff/cfg['# test ep'])
        a2test_rs.append(a2tr/cfg['# test ep'])
        a2test_eff.append(a2teff/cfg['# test ep'])

        results.append({
            'a1 train': a1train_rs,
            'a1 test': a1test_rs,
            'a1 eff': a1test_eff,
            'a1 model name': agent1.cfg['name'],

            'a2 train': a2train_rs,
            'a2 test': a2test_rs,
            'a2 eff': a2test_eff,
            'a2 model name': agent2.cfg['name'],

            'name': cfg['name'],
            'var': cfg['var. param name']
        })

    if cfg['save path']:
        with open(cfg['save path']+'results.json', 'w') as f:
            json.dump(results, f)

    return results