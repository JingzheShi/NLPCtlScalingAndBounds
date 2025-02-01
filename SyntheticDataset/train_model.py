import os
import torch
from argparse import ArgumentParser
from tqdm import tqdm

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = torch.nn.Linear(hidden_size//2, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.LeakyReLU(0.0015)
    def forward(self, task, ctrl):
        x = torch.cat([task, ctrl], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
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
        
        # y = task_output
        # task_output = self.linear_2_2(task_output)
        # task_output = self.relu(task_output)
        # task_output = self.linear_2_3(task_output)
        # task_output = y + task_output
        
        next_input = torch.cat([task_output, ctrl], dim=-1)
        next_input = self.linear_3(next_input)
        next_input = self.relu(next_input)
        return self.linear_4(next_input)

class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, task_bits, ctrl_bits, answers):
        self.task_bits = task_bits.float()
        self.ctrl_bits = ctrl_bits.float()
        self.answers = answers.float()
        print("Current length of task_bits: ", len(self.task_bits))
        self.length = len(self.task_bits)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.task_bits[idx], self.ctrl_bits[idx], self.answers[idx]

def train_model(model, train_loader, valid_loader, num_epochs, criterion, optimizer, save_position = None, early_stopping = -1):
    min_val_loss = float('inf')
    print("Save last.")
    no_improve_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        for task, ctrl, answer in train_loader:
            task = task.to(model.fc1.weight.device)
            ctrl = ctrl.to(model.fc1.weight.device)
            answer = answer.to(model.fc1.weight.device)
            optimizer.zero_grad()
            output = model(task, ctrl)
            output = torch.nn.functional.sigmoid(output).squeeze()
            loss = criterion(output, answer)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for task, ctrl, answer in valid_loader:
                task = task.to(model.fc1.weight.device)
                ctrl = ctrl.to(model.fc1.weight.device)
                answer = answer.to(model.fc1.weight.device)
                output = model(task, ctrl)
                output = torch.nn.functional.sigmoid(output).squeeze()
                output[output<0.000001] = output[output<0.000001] * 0 + 0.000001
                output[output>0.999999] = output[output>0.999999] * 0 + 0.999999
                loss = criterion(output, answer)
                valid_loss += loss.item()
        if valid_loss < min_val_loss:
            no_improve_epoch = 0
            min_val_loss = valid_loss
            if save_position is not None:
                torch.save(model.state_dict(), save_position)
            print(f"Ep {epoch+1}/{num_epochs}: Trn Loss: {train_loss/len(train_loader):.6f}, Val Loss: {valid_loss/len(valid_loader):.6f}. Reach Min Val Loss.")
        else:
            no_improve_epoch += 1
            print(f"Ep {epoch+1}/{num_epochs}: Trn Loss: {train_loss/len(train_loader):.6f}, Val Loss: {valid_loss/len(valid_loader):.6f}.")
        if early_stopping > 0 and no_improve_epoch >= early_stopping:
            print("Did not improve for ", early_stopping, " epochs. Early Stopping!")
            break
    print("Training Finished!")
    torch.save(model.state_dict(), save_position+'_last.pt')

def main():
    parser = ArgumentParser()
    
    # === basic settings ===
    parser.add_argument("--training_data_position", type=str, default='/root/autodl-tmp/working_dir/data/train.pt')
    parser.add_argument("--validation_data_position", type=str, default='/root/autodl-tmp/working_dir/data/valid.pt')
    
    # === basic settings ===
    parser.add_argument("--context_length", type=int, default=72) # 400 is the max context length set in generate_dataset.py
    parser.add_argument("--train_dataset_size", type=int, default=10000000) # 1000000 is the max train dataset size set in generate_dataset.py
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
    
    # === training parameters ===
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--early_stopping", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-3) # 010505_100: 1e-4; 010507_100: 2e-4; 010508_100: 2e-4
    parser.add_argument("--weight_decay", type=float, default=1e-4) # 010505_100: 1e-5; 010507_100: 2e-5; 010508_100: 1e-5
    parser.add_argument("--save_position", type=str, default='/root/autodl-tmp/working_dir/save_position_one__MLP_unslimited_corrected_3kk/')
    
    parser.add_argument("--device",type=str,default='cuda:0')
    
    args = parser.parse_args()
    
    
    
    train_data = torch.load(args.training_data_position)
    valid_data = torch.load(args.validation_data_position)
    
    train_task_bits = train_data['task_bits'][:args.train_dataset_size, :args.context_length]
    train_ctrl_bits = train_data['ctrl_bits'][:args.train_dataset_size, :]
    train_answers = train_data['answers'][:args.train_dataset_size]
    
    valid_task_bits = valid_data['task_bits'][:args.valid_dataset_size, :args.context_length]
    valid_ctrl_bits = valid_data['ctrl_bits'][:args.valid_dataset_size, :]
    valid_answers = valid_data['answers'][:args.valid_dataset_size]
    
    # only keep idx%200 == 0,1,...,50.
    # train_task_bits = train_task_bits[::200] is not correct. it will only keep idx%200 == 0.
    # train_task_bits = train_task_bits[::4] 
    
    
    
    
    train_dataset = TaskDataset(train_task_bits, train_ctrl_bits, train_answers)
    valid_dataset = TaskDataset(valid_task_bits, valid_ctrl_bits, valid_answers)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12)
    print("Dataset Loaded!")
    if args.use_bi_mlp:
        model = Bi_Input_MLP(args.context_length, len(train_ctrl_bits[0]), args.task_hidden_size, args.next_size, args.next_hidden_size, args.output_size)
    else:
        model = MLP(args.context_length + len(train_ctrl_bits[0]), args.hidden_size, args.output_size)
    
    model = model.to(torch.device(args.device))
    
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    os.makedirs(args.save_position, exist_ok=True)
    train_model(model, train_loader, valid_loader, args.num_epochs, loss_fn, optimizer, os.path.join(args.save_position,f'{args.hidden_size}hid_{args.context_length}ctl.pt'), args.early_stopping)

if __name__ == "__main__":
    main()