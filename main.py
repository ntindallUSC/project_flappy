from trainer import Trainer
from agent import Agent
from env import FlappyBird
import torch
import argparse

parser = argparse.ArgumentParser(description='Flappy-bird Configuration')
parser.add_argument('--mode', dest='mode', default='train', type=str)
parser.add_argument('--ckpt', dest='ckpt', default='none', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    agent = Agent()
    if args.mode == 'train':
        env = FlappyBird(record_every_episode=100, outdir='results/result/')
        tr = Trainer(agent, env)
        if args.ckpt != 'none':
            tr.load(args.ckpt, device)
        tr.train(device=device)
    else:
        env = FlappyBird(record_every_episode=1, outdir='eval/')
        tr = Trainer(agent, env)
        tr.load(args.ckpt, device)
        accumulated_reward, step = tr.run(device=device, explore=False)
        print('Accumulated_reward: {}, alive time: {}'.format(accumulated_reward, step))

