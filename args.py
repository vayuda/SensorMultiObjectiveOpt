import argparse
import json
import os
from env import GridWorld
from model_loader import create_agent, create_user
def get_config():
    # vary these parameters
    var_param_vals = [0.3,0.6,0.9,1]
    var_param_name = 'w2'
    args = parse_arguments()

    

    

    cfg = {
        'var. param name': var_param_name,
        'var. param vals': var_param_vals,

        #environment specs
        'trajectory length': args.trajectory_length,
        'coverage window': args.coverage_window,
        'exact position': args.exact_position,
        'w1': args.w1,
        'w2': args.w2,
        'start': tuple(map(int, args.start.split(','))),
        'goal': tuple(map(int, args.goal.split(','))),
        'grid size': args.grid_size,

        # training specs
        '# train ep': args.train_ep,
        '# test ep': args.test_ep,
        'ep len': args.ep_len,
        'name': args.exp_name,
        'save path': 'results/' + args.exp_name + '/' if args.save_result else None,

        #user specs
        'user wait time': args.user_wait_time,
        'user move probability': args.user_move_probability,
        'user model': args.user_model
    }

    cfg['test every'] = cfg['# train ep'] // 20

    

    if cfg['save path']:
        #if save path folder does not exist, create it
        if not os.path.exists(cfg['save path']):
            os.makedirs(cfg['save path'])
        with open(cfg['save path']+ 'config.json', 'w+') as f:
            json.dump(cfg, f)

    cfg['env'] = GridWorld({
        'trajectory length': args.trajectory_length,
        'coverage window': args.coverage_window,
        'exact position': args.exact_position,
        'w1': args.w1,
        'w2': args.w2,
        'start': tuple(map(int, args.start.split(','))),
        'goal': tuple(map(int, args.goal.split(','))),
        'grid size': args.grid_size
    })
    cfg['user'] = create_user(args.user, cfg)
    cfg['agent1'] = create_agent(args.agent1, cfg)
    cfg['agent2'] = create_agent(args.agent2, cfg)

    return cfg

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Configuration')
    
    #agent selection
    parser.add_argument('--agent1', type=str, default='SimpleAgent', help='Agent 1 name')
    parser.add_argument('--agent2', type=str, default='SensorQAgent', help='Agent 2 name')
    parser.add_argument('--user', type=str, default='CircleUser', help='User model name')

    #training
    parser.add_argument('--train_ep', type=int, default=20000, help='Number of training episodes')
    parser.add_argument('--test_ep', type=int, default=50, help='Number of testing episodes')
    parser.add_argument('--ep_len', type=int, default=100, help='Length of each episode')
    parser.add_argument('--exp_name', type=str, default='results', help='If saving results, name of results folder')
    parser.add_argument('--save_result', type=bool, default=False, help='Save results flag')

    #environment
    parser.add_argument('--trajectory_length', type=int, default=10, help='Number of positions in the trajectory (5 positions of size 2)')
    parser.add_argument('--coverage_window', type=int, default=16, help='Coverage window size')
    parser.add_argument('--exact_position', type=bool, default=False, help='Exact position flag')
    parser.add_argument('--w1', type=float, default=1, help='Weight 1')
    parser.add_argument('--w2', type=float, default=0.3, help='Weight 2')
    parser.add_argument('--start', type=str, default='0,0', help='Starting position (format: "x,y")')
    parser.add_argument('--goal', type=str, default='0,0', help='Goal position (format: "x,y")')
    parser.add_argument('--grid_size', type=int, default=5, help='Grid size')

    #user
    parser.add_argument('--user_wait_time', type=int, default=3, help='How long the user waits in the corner before moving')
    parser.add_argument('--user_move_probability', type=float, default=0.5, help='Probability of the user moving while in the corner of the board')
    parser.add_argument('--user_model', type=str, default='CircleUser', help='Model name')

    return parser.parse_args()

