import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--config_path', type=str, default='/scratch/project_xxxxxxxxxxx/NLPContextScaling/nanoGPT/config/')
parser.add_argument('--context_length', type=int, default=384)
parser.add_argument('--start_from',type=str,default='0.0')
parser.add_argument('--percent_name',type=str,default='0p1')
parser.add_argument('--long_context',type=bool,default=True)
args = parser.parse_args()
CONTEXT_LENGTH = args.context_length
PERCENT_NAME = args.percent_name
START_FROM = args.start_from

percent_name_to_str_dict = {
    '0p1':'0.1', # 9B*0.1% = 9M tokens, ~ 18 steps per epoch.
    '0p5':'0.5',
    '1p0':'1.0',
    '2p0':'2.0',
    '4p0':'4.0',
    '6p0':'6.0',
    '10p0':'10.0',
}

file_content = r'''
# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
n_layer = 6
wandb_project = 'NLP_Long_contextScaling_071403ver'''+r'''pct_'''+r'''_ctl_fp32_6layer'
wandb_run_name='gpt2-6layer_01+'''+PERCENT_NAME+r'''pct_'''+str(CONTEXT_LENGTH)+r'''ctl'

# these make the total batch size be ~0.5M
# 32 batch size * 384 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = int(32 * 384 / '''+ str(CONTEXT_LENGTH) + r''')
# block_size =  # context_length
block_size = ''' + str(CONTEXT_LENGTH) + r'''
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff, eval_interval is how often to run eval, currently set to 100 * 0.5M = 50M tokens.
eval_interval = 45
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# 9B * 0.5% = 45M tokens
train_file_name = 'LONGCONTEXTSUBSET_train_P'''+percent_name_to_str_dict[PERCENT_NAME]+r'''_SFrom'''+START_FROM+r'''.bin'
'''

# write to file.
config_file_path = os.path.join(args.config_path,'LONGCONTEXT_train_gpt2_'+str(CONTEXT_LENGTH)+'ctl_'+PERCENT_NAME+'pct_small.py')

with open(config_file_path, 'w') as f:
    f.write(file_content)
    print(f'File written to {config_file_path}')
