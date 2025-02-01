from train_model import MLP, Bi_Input_MLP, TaskDataset, train_model
from argparse import ArgumentParser
from tqdm import tqdm
import torch

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = torch.nn.Linear(hidden_size//2, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.LeakyReLU(0.005)
    def forward(self, task, ctrl):
        x = torch.cat([task, ctrl], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

class Bi_Input_MLP(torch.nn.Module):
    def __init__(self, task_bits, ctrl_bits, task_hidden_size, next_size, next_hidden_size, output_size):
        super(Bi_Input_MLP, self).__init__()
        self.fc1 = torch.nn.Linear(task_bits, task_hidden_size)
        self.linear_2 = torch.nn.Linear(task_hidden_size, next_size)
        self.linear_2_2 = torch.nn.Linear(next_size, task_hidden_size)
        self.linear_2_3 = torch.nn.Linear(task_hidden_size, next_size)
        self.linear_3 = torch.nn.Linear(next_size + ctrl_bits, next_hidden_size)
        self.linear_4 = torch.nn.Linear(next_hidden_size, output_size)
        self.relu = torch.nn.LeakyReLU(0.001)
        
    def forward(self, task, ctrl):
        task_output = self.fc1(task)
        task_output = self.relu(task_output)
        task_output = self.linear_2(task_output)
        return task_output
        
        # y = task_output
        # task_output = self.linear_2_2(task_output)
        # task_output = self.relu(task_output)
        # task_output = self.linear_2_3(task_output)
        # task_output = y + task_output
        
        next_input = torch.cat([task_output, ctrl], dim=-1)
        next_input = self.linear_3(next_input)
        next_input = self.relu(next_input)
        return self.linear_4(next_input)

def save_mid_feature(model, dataset, save_position):
    model.eval()
    outputs = list()
    with torch.no_grad():
        for task, ctrl, answer in tqdm(dataset):
            task = task.to(model.fc1.weight.device)
            ctrl = ctrl.to(model.fc1.weight.device)
            answer = answer.to(model.fc1.weight.device)
            output = model(task, ctrl).detach().cpu()
            outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    torch.save(outputs, save_position)

def main():
    parser = ArgumentParser()
    
    parser.add_argument('--save_position', type=str, default='/root/autodl-tmp/working_dir/mid_feature/mid_feature_100ctl.pt')
    
    parser.add_argument("--model_state_dict_position", type=str, default='/root/autodl-tmp/working_dir/save_position/100hid_100ctl.pt')
    
    parser.add_argument("--validation_data_position", type=str, default='/root/autodl-tmp/working_dir/data/valid.pt')
    parser.add_argument("--context_length", type=int, default=100) # 400 is the max context length set in generate_dataset.py
    
    parser.add_argument("--visible_context_length", type=int, default=-1)
    parser.add_argument("--valid_dataset_size", type=int, default=2000000) # 100000 is the max valid dataset size set in generate_dataset.py
    
    # === model parameters ===
    # model selection
    parser.add_argument("--use_bi_mlp", action = 'store_true')
    # for bi mlp:
    parser.add_argument("--task_hidden_size", type=int, default=400)
    parser.add_argument("--next_size", type=int, default=80)
    parser.add_argument("--next_hidden_size", type=int, default=200)
    
    # for mlp:
    parser.add_argument("--hidden_size", type=int, default=400)
    # for both:
    parser.add_argument("--output_size", type=int, default=1)
    
    parser.add_argument("--batch_size", type=int, default=10000)
    
    
    args = parser.parse_args()


    valid_data = torch.load(args.validation_data_position)
    valid_task_bits = valid_data['task_bits'][:args.valid_dataset_size, :args.context_length]
    
    if args.visible_context_length != -1:
        valid_task_bits[:, args.visible_context_length:] = valid_task_bits[:, args.visible_context_length:]*0 + 0.5
    
    valid_ctrl_bits = valid_data['ctrl_bits'][:args.valid_dataset_size, :]
    valid_answers = valid_data['answers'][:args.valid_dataset_size]
    valid_dataset = TaskDataset(valid_task_bits, valid_ctrl_bits, valid_answers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    
    if args.use_bi_mlp:
        model = Bi_Input_MLP(args.context_length, len(valid_ctrl_bits[0]), args.task_hidden_size, args.next_size, args.next_hidden_size, args.output_size)
    else:
        model = MLP(args.context_length + len(valid_ctrl_bits[0]), args.hidden_size, args.output_size)
    
    model.load_state_dict(torch.load(args.model_state_dict_position))
    
    save_mid_feature(model, valid_loader, args.save_position)
    
if __name__ == '__main__':
    main()