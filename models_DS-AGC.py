import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import adv_layer
import random
from self_attention import MultiHeadAttention
import scipy.spatial.distance as dis
def diff_loss(diff, S, Falpha):
    '''
    compute the 1st loss of L_{graph_learning}
    '''
    if len(S.shape)==4:
        # batch input
        return Falpha * torch.mean(torch.sum(torch.sum(diff**2,axis=3)*S, axis=(1,2)))
    else:
        return Falpha * torch.sum(torch.matmul(S,torch.sum(diff**2,axis=2)))
  
def F_norm_loss(S, Falpha):
    '''
    compute the 2nd loss of L_{graph_learning}
    '''
    if len(S.shape)==3:
        # batch input
        return Falpha * torch.sum(torch.mean(S**2,axis=0))
    else:
        return Falpha * torch.sum(S**2)


class Graph_Learn(nn.Module):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self,alpha, num_of_features, device):
        super(Graph_Learn, self).__init__()
        self.alpha = alpha
        self.a = nn.init.ones_(nn.Parameter(torch.FloatTensor(num_of_features, 1).to(device)))
    def forward(self, x):
        N, V, f = x.shape
        diff = (x.expand(V, N, V, f).permute(2, 1, 0, 3)-x.expand(V, N, V, f)).permute(1, 0, 2, 3)#62*61+62
        tmpS = torch.exp(-F.relu(torch.reshape(torch.matmul(torch.abs(diff), self.a), [N, V, V])))
        S = tmpS / torch.sum(tmpS, axis=1, keepdims=True)
        Sloss = F_norm_loss(S, 1)
        dloss = diff_loss(diff, S, self.alpha)
        ajloss = Sloss + dloss
        return S, ajloss


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_vertices, num_of_filters)
    '''
    def __init__(self, num_of_filters, k, num_of_features, device):
        super(cheb_conv, self).__init__()
        self.Theta = nn.ParameterList([nn.init.uniform_(nn.Parameter(torch.FloatTensor(num_of_features, num_of_filters).to(device))) for _ in range(k)])
        self.out_channels = num_of_filters
        self.K = k
        self.device = device
        
    def forward(self, x):
        assert isinstance(x, list)
        x, W = x
        N, V, f = x.shape
        #Calculating Chebyshev polynomials
        D = torch.diag_embed(torch.sum(W,axis=1))
        L = D - W
        '''
        Here we approximate λ_{max} to 2 to simplify the calculation.
        For more general calculations, please refer to here:
            lambda_max = K.max(tf.self_adjoint_eigvals(L),axis=1)
            L_t = (2 * L) / tf.reshape(lambda_max,[-1,1,1]) - [tf.eye(int(num_of_vertices))]
        '''
        lambda_max = 2.0
        L_t =( (2 * L) / lambda_max - torch.eye(int(V)).to(self.device))
        cheb_polynomials = [torch.eye(int(V)).to(self.device), L_t]
        for i in range(2, self.K):
            cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        #Graph Convolution
        outputs = []
        graph_signal = x # (b, V, F_in)
        output = torch.zeros(N, V, self.out_channels).to(self.device)  # (b, V, F_out)
        for k in range(self.K):
            T_k = cheb_polynomials[k]  # (V,V)
            theta_k = self.Theta[k]  # (in_channel, out_channel)
            rhs = T_k.matmul(graph_signal)
            output = output + rhs.matmul(theta_k)  # (b, V, F_in)(F_in, F_out) = (b, V, F_out)
        outputs.append(output)  # (b, V, F_out)
        return F.relu(torch.cat(outputs, dim=1))  # (b, V, F_out)

class GCN_block(nn.Module):

    def __init__(self, net_params):
        super(GCN_block, self).__init__()
        self.num_of_features = net_params['num_of_features']
        device = net_params['DEVICE']
        node_feature_hidden1 = net_params['node_feature_hidden1']
        self.Graph_Learn = Graph_Learn(net_params['GLalpha'], self.num_of_features, device)
        self.cheb_conv = cheb_conv(node_feature_hidden1,net_params['K'], self.num_of_features, device)

    def forward(self, x):
        S, ajloss = self.Graph_Learn(x)
        gcn = self.cheb_conv([x, S])
        return gcn, S, ajloss

class feature_extractor(nn.Module):
    def __init__(self,input, hidden_1, hidden_2):
         super(feature_extractor,self).__init__()
         self.fc1=nn.Linear(input,hidden_1)
         self.fc2=nn.Linear(hidden_1,hidden_2)
         self.dropout1 = nn.Dropout(p=0.25)
         self.dropout2 = nn.Dropout(p=0.25)
    def forward(self,x):
         x1=self.fc1(x)
         x1=F.relu(x1)
#         x=F.leaky_relu(x)
         x2=self.fc2(x1)
         x2=F.relu(x2)
#         x=F.leaky_relu(x)
         return x1, x2


def aug_drop_node_list(graph_list, drop_percent):
    graph_num = len(graph_list)  # number of graphs
    aug_list = []
    for i in range(graph_num):
        # aug_graph = aug_drop_node((graph_list[i]), drop_percent)
        aug_graph = aug_selet_node((graph_list[i]), drop_percent)
        aug_list.append(aug_graph)
    aug = torch.stack(aug_list, 0)
    aug = torch.flatten(aug, start_dim=1, end_dim=-1)
    return aug

def aug_selet_node(graph, drop_percent=0.8):
    num = len(graph)  # number of nodes of one graph
    selet_num = int(num * drop_percent)  # number of drop nodes
    aug_graph = graph.clone()
    all_node_list = [i for i in range(num)]
    selet_node_list = random.sample(all_node_list, selet_num)
    aug_graph = torch.index_select(aug_graph, 0, torch.IntTensor(selet_node_list).cuda())
    return aug_graph

def aug_drop_node(graph, drop_percent=0.2):
    num = len(graph)  # number of nodes of one graph
    drop_num = int(num * drop_percent)  # number of drop nodes
    aug_graph = graph.clone()
    all_node_list = [i for i in range(num)]
    drop_node_list = random.sample(all_node_list, drop_num)
    aug_graph[drop_node_list] = 0
    return aug_graph

class projection_head(nn.Module):

    def __init__(self, input_dim, output_dim):  # L=nb_hidden_layers
        super().__init__()
        self.fc_layer1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc_layer2 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        x = self.fc_layer1(x)
        x = F.relu(x)
        x = self.fc_layer2(x)
        return x

class SemiGCL(nn.Module):
    def __init__(self, net_params):
        super(SemiGCL, self).__init__()
        self.device = net_params['DEVICE']
        out_feature = net_params['node_feature_hidden1']
        channel = net_params['num_of_vertices']
        linearsize = net_params['linearsize']
        self.drop_rate = net_params['drop_rate']
        self.GCN = GCN_block(net_params)
        self.domain_classifier2 = adv_layer.DomainAdversarialLoss(hidden_1=64)
        self.domain_classifier3 = adv_layer.TripleDomainAdversarialLoss(hidden_1=64)
        self.domain_classifier2_3 = adv_layer.DomainAdversarialLoss_threeada(hidden_1=64)
        self.fea_extrator_f = feature_extractor(310, 64, 64)
        self.fea_extrator_g = feature_extractor(int(channel*self.drop_rate) * out_feature, linearsize, 64)
        self.fea_extrator_c = feature_extractor(64*2, 64, 32)
        self.projection_head = projection_head(64, 16)
        self.classifier = nn.Linear(64, net_params['category_number'])
        self.self_attention=MultiHeadAttention(128, 128, 128, 128, 64, 0.5)
        self.batch_size = net_params['batch_size']
        self.category_number = net_params['category_number']
        self.Multi_att = net_params['Multi_att']
    def forward(self, x, tripleada=0, threshold=0):

        feature, S, ajloss = self.GCN(x)
        feature1 = torch.flatten(x, start_dim=1, end_dim=-1)
        _, feature1 = self.fea_extrator_f(feature1)
        if threshold:
            if tripleada:
                domain_output = self.domain_classifier3(feature1)
            else:
                domain_output = self.domain_classifier2_3(feature1)
        else:
            domain_output = self.domain_classifier2(feature1)
        aug_graph1, aug_graph2 = aug_drop_node_list(feature, self.drop_rate), aug_drop_node_list(feature, self.drop_rate)
        _, aug_graph1_feature1 = self.fea_extrator_g(aug_graph1)
        _, aug_graph2_feature1 = self.fea_extrator_g(aug_graph2)

        aug_graph1_feature = self.projection_head(aug_graph1_feature1)
        aug_graph2_feature = self.projection_head(aug_graph2_feature1)

        L2 = torch.mean((aug_graph1_feature1 - aug_graph2_feature1) ** 2)

        sim_matrix_tmp2 = self.sim_matrix2(aug_graph1_feature, aug_graph2_feature, temp=1)
        row_softmax = nn.LogSoftmax(dim=1)
        row_softmax_matrix = -row_softmax(sim_matrix_tmp2)

        colomn_softmax = nn.LogSoftmax(dim=0)
        colomn_softmax_matrix = -colomn_softmax(sim_matrix_tmp2)

        row_diag_sum = self.compute_diag_sum(row_softmax_matrix)
        colomn_diag_sum = self.compute_diag_sum(colomn_softmax_matrix)
        contrastive_loss = (row_diag_sum + colomn_diag_sum) / (2 * len(row_softmax_matrix))

        class_feature = torch.cat((feature1, aug_graph1_feature1), dim=1)

        class_feature = class_feature.unsqueeze(1)
        if self.Multi_att:
            class_feature = self.self_attention(class_feature, class_feature, class_feature)
        class_feature = class_feature.squeeze(1)
        class_feature, _ = self.fea_extrator_c(class_feature)
        pred = self.classifier(class_feature)

        s_feature = class_feature[:self.batch_size]
        t_feature = class_feature[-self.batch_size:]

        sim_sample = self.sim_matrix2(s_feature, t_feature)
        sim_weight = torch.mean(sim_sample, dim=1).unsqueeze(1)
        sim_weight = torch.nn.functional.softmax(sim_weight, dim=0)

        return pred, domain_output, ajloss, contrastive_loss, sim_weight, L2


    def sharpen(self, predict, t):
        e = torch.sum((predict) ** (1 / t), dim=1).unsqueeze(dim=1)
        predict = (predict ** (1 / t)) / e.expand(len(predict), self.category_number)
        return predict

    def compute_diag_sum(self, tensor):
        num = len(tensor)
        diag_sum = 0
        for i in range(num):
            diag_sum += tensor[i][i]
        return diag_sum

    def sim_matrix2(self, ori_vector, arg_vector, temp=1.0):
        for i in range(len(ori_vector)):
            sim = torch.cosine_similarity(ori_vector[i].unsqueeze(0), arg_vector, dim=1) * (1 / temp)
            if i == 0:
                sim_tensor = sim.unsqueeze(0)
            else:
                sim_tensor = torch.cat((sim_tensor, sim.unsqueeze(0)), 0)
        return sim_tensor


    def sim_matrix(self, s_vector, t_vector):
        pdist = nn.PairwiseDistance(p=1)#p=2就是计算欧氏距离，p=1就是曼哈顿距离
        for i in range(len(s_vector)):
            sim = pdist(s_vector[i].unsqueeze(0).repeat(len(s_vector),1), t_vector)
            if i == 0:
                sim_tensor = sim.unsqueeze(0)
            else:
                sim_tensor = torch.cat((sim_tensor, sim.unsqueeze(0)), 0)
        sim_tensor = torch.exp(-sim_tensor)
        return sim_tensor.to(self.device)

    def predict(self, x):
        self.eval()
        feature, _, _ = self.GCN(x)
        feature1 = torch.flatten(x, start_dim=1, end_dim=-1)
        _, feature1 = self.fea_extrator_f(feature1)
        aug_graph1 = aug_drop_node_list(feature, self.drop_rate)
        _, aug_graph1_feature1 = self.fea_extrator_g(aug_graph1)
        class_feature = torch.cat((feature1, aug_graph1_feature1), dim=1)
        class_feature = class_feature.unsqueeze(1)
        if self.Multi_att:
            class_feature = self.self_attention(class_feature, class_feature, class_feature)
        class_feature = class_feature.squeeze(1)
        class_feature, _ = self.fea_extrator_c(class_feature)
        pred = self.classifier(class_feature)
        label_feature = torch.nn.functional.softmax(pred, dim=1)
        return label_feature