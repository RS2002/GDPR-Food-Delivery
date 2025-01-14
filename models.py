import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes=[64,64,64,1], arl=False, dropout=0.0):
        super().__init__()
        self.arl = arl
        self.attention = nn.Sequential(
            nn.Linear(layer_sizes[0],layer_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes[0],layer_sizes[0])
        )

        self.layer_sizes = layer_sizes
        if len(layer_sizes) < 2:
            raise ValueError()
        self.layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        if self.arl:
            x = x * self.attention(x)
        for layer in self.layers[:-1]:
            x = self.dropout(self.act(layer(x)))
        x = self.layers[-1](x)
        return x

class LSTM(nn.Module):
    def __init__(self,input_size, output_size, hidden_size=64, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = MLP([hidden_size, hidden_size, output_size], dropout=dropout)

    def forward(self,x,valid_index=None):
        self.lstm.flatten_parameters()
        x,_ = self.lstm(x)
        x = torch.concat([torch.zeros(x.shape[0],1,x.shape[2]).to(x.device),x],dim=1)
        if valid_index is not None:
            x = x[torch.arange(x.size(0)),valid_index]
        else:
            x = x[:, -1, :]
        x = self.fc(x)
        return x

class BiLSTM(nn.Module):
    def __init__(self,input_size, output_size, hidden_size=64, dropout=0.0):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False, dropout=dropout)
        self.lstm2 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = MLP([hidden_size * 2, hidden_size, output_size], dropout=dropout)


    def forward(self,x,valid_index=None):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        x1,_ = self.lstm1(x)
        x1 = torch.concat([torch.zeros(x1.shape[0],1,x1.shape[2]).to(x1.device),x1],dim=1)
        if valid_index is not None:
            x1 = x1[torch.arange(x1.size(0)),valid_index]
            x2 = x.clone()
            for i in range(x2.shape[0]):
                end_index = valid_index[i].item()
                x2[i, :end_index] = x[i, :end_index].flip(dims=(0,))
        else:
            x1 = x1[:, -1, :]
            x2 = torch.flip(x, [1])
        x2, _ = self.lstm2(x2)
        x2 = torch.concat([torch.zeros(x2.shape[0],1,x2.shape[2]).to(x2.device),x2],dim=1)
        if valid_index is not None:
            x2 = x2[torch.arange(x2.size(0)),valid_index]
        else:
            x2 = x2[:, -1, :]
        x = torch.cat((x1, x2), dim=-1)
        x = self.fc(x)
        return x

class Worker_Net(nn.Module):
    def __init__(self, state_size=7, order_size=4, output_dim=32, bi_direction=False, dropout=0.0):
        super().__init__()
        if bi_direction:
            self.lstm = BiLSTM(order_size, output_dim, dropout=dropout)
        else:
            self.lstm = LSTM(order_size, output_dim, dropout=dropout)
        self.encode = MLP([state_size, output_dim, output_dim], arl=True, dropout=dropout)
        self.mlp = MLP([output_dim * 2, output_dim, output_dim], dropout=dropout)

    def forward(self,x_state,x_order,order_num=None):
        x_order = self.lstm(x_order,order_num)
        x_state = self.encode(x_state)
        y = self.mlp(torch.concat([x_state,x_order],dim=-1))
        return y

# class Order_Net(nn.Module):
#     def __init__(self, state_size=5, output_size=32, dropout=0.0):
#         super().__init__()
#         self.model = MLP([state_size,16,32,output_size], arl=True, dropout=dropout)
#
#     def forward(self,x):
#         y = self.model(x)
#         return y

class Order_Net(nn.Module):
    def __init__(self, state_size=5, output_size=32, dropout=0.0):
        super().__init__()
        self.order_encoder = MLP([state_size - 11, output_size, output_size], arl=True, dropout=dropout)
        self.global_encoder = MLP([11, output_size, output_size], arl=True, dropout=dropout)
        self.fusion_layer = MLP([output_size * 2, output_size, output_size], arl=False, dropout=dropout)

    def forward(self, x):
        y1 = self.order_encoder(x[:,:-11])
        y2 = self.global_encoder(x[:,-11:])
        y = torch.concat([y1,y2],dim=-1)
        y = self.fusion_layer(y)
        return y

class Attention_Score(nn.Module):
    def __init__(self, input_dims=64, hidden_dims=64, head=1, dropout=0.0):
        super().__init__()
        self.q_linear = nn.ModuleList()
        self.k_linear = nn.ModuleList()
        for i in range(int(head)):
            self.q_linear.append(MLP([input_dims,hidden_dims,hidden_dims], dropout=dropout))
            self.k_linear.append(MLP([input_dims,hidden_dims,hidden_dims], dropout=dropout))
        self.head = head
        if self.head != 1:
            self.fuse_layer = MLP([head,16,1], dropout=dropout)

    def forward(self,q,k):
        attn_matrix = None

        for i in range(self.head):
            query=self.q_linear[i](q)
            key=self.k_linear[i](k)
            attn = torch.mm(query,key.T)
            attn = attn.unsqueeze(-1)
            if attn_matrix is None:
                attn_matrix = attn
            else:
                attn_matrix = torch.concat([attn_matrix, attn],dim=-1)

        # attn_matrix = torch.mean(attn_matrix, dim=-1)

        if self.head != 1:
            attn_matrix = self.fuse_layer(attn_matrix)
        attn_matrix = attn_matrix.squeeze(-1)
        return attn_matrix

# class Attention_Score(nn.Module):
#     def __init__(self, input_dims=64,hidden_dims=64,head=1, dropout=0.0):
#         super().__init__()
#         self.q_linear = MLP([input_dims,hidden_dims,hidden_dims], dropout=dropout)
#         self.k_linear = MLP([input_dims,hidden_dims,hidden_dims], dropout=dropout)
#         self.head=head
#         self.num=hidden_dims//head
#         self.input_dims=input_dims
#
#
#     def forward(self,q,k):
#         query=self.q_linear(q)
#         key=self.k_linear(k)
#
#         query=query.view(-1,self.head,self.num).transpose(0, 1)
#         key=key.view(-1,self.head,self.num).transpose(0, 1)
#
#         attn_matrix=torch.bmm(query,key.transpose(1, 2))
#         attn_matrix=torch.sum(attn_matrix,dim=0)
#
#         return attn_matrix

class Q_Net(nn.Module):
    def __init__(self, state_size=7, history_order_size=4, current_order_size=5, hidden_dim=64, head=1, bi_direction=False, dropout=0.0):
        super().__init__()
        self.worker_net = Worker_Net(state_size=state_size, order_size=history_order_size, output_dim=hidden_dim, bi_direction=bi_direction, dropout=dropout)
        self.order_net = Order_Net(state_size=current_order_size, output_size=hidden_dim, dropout=dropout)
        self.mask_net = Worker_Net(state_size=state_size-4, order_size=history_order_size-1, output_dim=hidden_dim, bi_direction=bi_direction, dropout=dropout)
        self.attention = Attention_Score(input_dims=hidden_dim,hidden_dims=hidden_dim,head=head, dropout=dropout)
        self.sigmoid = nn.Sigmoid()


    def forward(self,order,x_state,x_order,order_num=None,mask=None):
        order_num = order_num.int()
        order = order.float()
        x_state = x_state.float()
        x_order = x_order.float()

        order = self.order_net(order)
        worker = self.worker_net(x_state,x_order,order_num)
        if mask is not None:
            mask = mask.clone()
            mask[mask>1]=1
            x_state = torch.concat([x_state[...,:-5],x_state[...,-1:]],dim=-1) # drop the history price
            x_order = x_order[...,:-1] # drop the order price
            worker_mask = self.mask_net(x_state,x_order,order_num)
            while len(mask.shape)<len(worker.shape):
                mask = mask.unsqueeze(-1)
            worker = worker * (1-mask) + worker_mask * mask
        q_matrix = self.attention(worker,order)

        return q_matrix



class Assignment_Net(nn.Module):
    def __init__(self, state_size=7, history_order_size=4, current_order_size=5, hidden_dim=64, head=1, bi_direction=False, dropout=0.0):
        super().__init__()
        self.worker_net = Worker_Net(state_size=state_size, order_size=history_order_size, output_dim=hidden_dim, bi_direction=bi_direction, dropout=dropout)
        self.order_net = Order_Net(state_size=current_order_size, output_size=hidden_dim, dropout=dropout)
        self.mask_net = Worker_Net(state_size=state_size-4, order_size=history_order_size-1, output_dim=hidden_dim, bi_direction=bi_direction, dropout=dropout)
        # Q-value
        self.attention = Attention_Score(input_dims=hidden_dim,hidden_dims=hidden_dim,head=head, dropout=dropout)
        # Payment
        self.attention_price_mu = Attention_Score(input_dims=hidden_dim, hidden_dims=hidden_dim, head=head, dropout=dropout)
        self.attention_price_sigma = Attention_Score(input_dims=hidden_dim, hidden_dims=hidden_dim, head=head, dropout=dropout)
        self.sigmoid = nn.Sigmoid()
        # Estimated Earning
        self.predictor = MLP([hidden_dim, hidden_dim, hidden_dim, 6])

        self.multi_lora = nn.ModuleList()
        for i in range(10):
            self.multi_lora.append(MLP([state_size-3,4,hidden_dim],arl=True,dropout=dropout))

        self.softplus = nn.Softplus()

    def encode(self,order,x_state,x_order,order_num=None,mask=None):
        order_num = order_num.int()
        order = order.float()
        x_state = x_state.float()
        x_order = x_order.float()
        order = self.order_net(order)
        worker = self.worker_net(x_state, x_order, order_num)
        if mask is not None:
            mask = mask.clone()
            mask[mask > 1] = 1
            x_state = torch.concat([x_state[..., :-5], x_state[..., -1:]], dim=-1)  # drop the history price
            x_order = x_order[..., :-1]  # drop the order price
            worker_mask = self.mask_net(x_state, x_order, order_num)
            while len(mask.shape) < len(worker.shape):
                mask = mask.unsqueeze(-1)
            worker = worker * (1 - mask) + worker_mask * mask
        return order, worker

    def forward(self,order,x_state,x_order,order_num=None,mask=None):
        order_emb, worker_emb = self.encode(order,x_state,x_order,order_num,mask)
        q_matrix = self.attention(worker_emb,order_emb)

        order_emb, worker_emb = order_emb.detach(), worker_emb.detach()
        price_mu_matrix = self.attention_price_mu(worker_emb, order_emb)
        price_mu_matrix = self.sigmoid(price_mu_matrix) * 2
        price_sigma_matrix = self.attention_price_sigma(worker_emb, order_emb)
        price_sigma_matrix = self.sigmoid(price_sigma_matrix) * 0.3 + 1e-5

        return q_matrix, price_mu_matrix, price_sigma_matrix

    def worker_encode(self,x_state):
        x_state = x_state.float()
        worker = self.worker_net.encode(x_state)

        x_state = torch.concat([x_state[..., :-5], x_state[..., -1:]], dim=-1)  # drop the history price
        worker_mask = self.mask_net.encode(x_state)

        worker_emb = torch.concat([worker, worker_mask], dim=-1)
        return worker_emb

    # def estimate_earning(self,x_state):
    #     # worker_emb = self.worker_encode(x_state)
    #     # worker_emb = worker_emb.detach()
    #     # earning = self.predictor(worker_emb)
    #
    #     x_state = x_state.float()
    #     worker = self.worker_net.encode(x_state)
    #     worker_emb = worker.detach()
    #     earning = self.predictor2(worker_emb)
    #
    #     earning = self.softplus(earning)
    #     earning = earning.reshape([-1,3,2])
    #
    #     return earning

    # def estimate_earning(self,x_state,x_private):
    #
    #     x_state = x_state.float()
    #     worker = self.worker_net.encode(x_state)
    #     worker = worker.detach()
    #
    #     x_private = self.private_encoder(x_private)
    #     worker_emb = torch.concat([worker,x_private],dim=-1)
    #
    #     earning = self.predictor(worker_emb)
    #
    #     earning = self.softplus(earning)
    #     earning = earning.reshape([-1,3,2])
    #
    #     return earning

    def estimate_earning(self,x_state,x_private):

        x_state = x_state.float()
        worker = self.worker_encode(x_state)
        worker = worker.detach()

        group_id = (x_private[-1] - 0.85) // 0.03
        if group_id < 0:
            group_id = 0
        elif group_id > 9:
            group_id = 9
        private_encoder = self.multi_lora[group_id]
        x_private = private_encoder(x_private)
        worker_emb = worker + x_private

        earning = self.predictor(worker_emb)

        earning = self.softplus(earning)
        earning = earning.reshape([-1,3,2])

        return earning


class Price_Net(nn.Module):
    def __init__(self, state_size=7, history_order_size=4, current_order_size=5, hidden_dim=64, head=1, bi_direction=False, dropout=0.0):
        super().__init__()
        self.worker_net = Worker_Net(state_size=state_size, order_size=history_order_size, output_dim=hidden_dim, bi_direction=bi_direction, dropout=dropout)
        self.order_net = Order_Net(state_size=current_order_size, output_size=hidden_dim, dropout=dropout)
        self.mask_net = Worker_Net(state_size=state_size-4, order_size=history_order_size-1, output_dim=hidden_dim, bi_direction=bi_direction, dropout=dropout)
        self.attention_price_mu = Attention_Score(input_dims=hidden_dim,hidden_dims=hidden_dim,head=head, dropout=dropout)
        self.attention_price_sigma = Attention_Score(input_dims=hidden_dim,hidden_dims=hidden_dim,head=head, dropout=dropout)
        self.sigmoid = nn.Sigmoid()


    def forward(self,order,x_state,x_order,order_num=None,mask=None):
        order_num = order_num.int()
        order = order.float()
        x_state = x_state.float()
        x_order = x_order.float()

        order = self.order_net(order)
        worker = self.worker_net(x_state,x_order,order_num)

        if mask is not None:
            mask = mask.clone()
            mask[mask>1]=1
            x_state = torch.concat([x_state[...,:-5],x_state[...,-1:]],dim=-1) # drop the history price
            x_order = x_order[...,:-1] # drop the order price
            worker_mask = self.mask_net(x_state,x_order,order_num)
            while len(mask.shape)<len(worker.shape):
                mask = mask.unsqueeze(-1)
            worker = worker * (1-mask) + worker_mask * mask

        price_mu_matrix = self.attention_price_mu(worker,order)
        price_mu_matrix = self.sigmoid(price_mu_matrix) * 2

        price_sigma_matrix = self.attention_price_sigma(worker,order)
        price_sigma_matrix = self.sigmoid(price_sigma_matrix) * 0.3 + 1e-5

        return price_mu_matrix,price_sigma_matrix

class Worker_Q_Net(nn.Module):
    def __init__(self, input_size=14, history_order_size=4, output_dim=2, bi_direction=False, dropout=0.0): # (7+1)+(5+1)=14
        super().__init__()
        self.worker_net = Worker_Net(state_size=input_size, order_size=history_order_size, output_dim=output_dim, bi_direction=bi_direction, dropout=dropout)

    def forward(self,x,x_order,order_num=None):
        order_num = order_num.int()
        x_order = x_order.float()
        x = x.float()
        y = self.worker_net(x,x_order,order_num) # [reject_prob,accept_prob]
        return y
