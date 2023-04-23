'''
Script to take encoders of different modalities and concatenate them. Delete the old paths after.
'''
import argparse
import torch
import os.path as osp
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--encoder_path', type=str, required=True, help='Path to encoders')
parser.add_argument('--num_modalities', type=int, required=True, help='Number of modalities')

if __name__ == '__main__':
    args = parser.parse_args()
    path = args.encoder_path
    all_paths = [sorted(glob(osp.join(path, f"encoder_*_*_{i}.pth"))) for i in range(args.num_modalities)]
    N = len(all_paths[0])
    assert(all([len(p) == N for p in all_paths])), "Number of encoders for each modality should be the same"
    print(N)
    # Concatenate them 
    for i in range(N):
        encoder_paths = [p[i] for p in all_paths]
        encoder_states = [torch.load(p) for p in encoder_paths]
        encoder_state = {}
        for k in ['embeddings']:
            encoder_state[k] = torch.cat([s[k] for s in encoder_states], dim=1)
        encoder_state['resolutions'] = encoder_states[0]['resolutions']
        encoder_state['offsets'] = encoder_states[0]['offsets']
        # get new path
        new_path = osp.join(path, encoder_paths[0].replace("_0.pth", ".pth"))
        # for k, v in encoder_state.items():
        #     print(k, v.shape)
        print("Saving {} to {}.".format(", ".join([x.split('/')[-1] for x in encoder_paths]), new_path))
        torch.save(encoder_state, new_path)
        for p in encoder_paths:
            os.remove(p)