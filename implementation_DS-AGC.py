'''
Description: 
Author: voicebeer
Date: 2020-09-14 01:01:51
LastEditTime: 2021-12-28 01:46:52
'''
import torch
import numpy as np
from tqdm import tqdm
import models
import os
import random
from torch.optim import RMSprop
from sklearn import preprocessing
import scipy.io as scio
import torch.utils.data as Data
from AutoWeight import AutomaticWeightedLoss
import torch.nn as nn
from torch.nn import init
sys_path = os.path.abspath("..")
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        setup_seed(20)
        init.xavier_uniform_(m.weight.data)##对参数进行xavier初始化，为了使得网络中信息更好的流动，每一层输出的方差应该尽量相等
        init.constant_(m.bias.data,0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        setup_seed(20)
        m.weight.data.normal_(0,0.03)
#        torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()

def get_cos_similarity_distance(pseudo, pred):
    """Get distance in cosine similarity
    :param features: features of samples, (batch_size, num_clusters)
    :return: distance matrix between features, (batch_size, batch_size)
    """
    pseudo_norm = torch.norm(pseudo, dim=1, keepdim=True)
    pseudo = pseudo / pseudo_norm

    pred_norm = torch.norm(pred, dim=1, keepdim=True)
    pred = pred / pred_norm

    cos_dist_matrix = torch.mm(pseudo, pred.transpose(0, 1))
    return cos_dist_matrix

def train_GCN_contrast(subject_id, parameter, net_params, source_labeled_loaders, source_unlabeled_loaders, target_loader):
    device=net_params['DEVICE']
    setup_seed(20)
    model=models.SemiGCL(net_params).to(device)
    setup_seed(20)
    model.apply(weigth_init)
    awl = AutomaticWeightedLoss(4)
    optimizer = RMSprop([{'params':model.parameters(), 'lr':parameter['init_lr'], 'weight_decay':parameter['weight_decay']},
                        {'params': awl.parameters(), 'weight_decay': 0}])
    best_acc, best_test_acc = 0.0, 0.0
    acc_list = np.zeros(parameter['epochs'])
    threshold = parameter['threshold']
    for epoch in range(parameter['epochs']):
        model.train()
        total_loss, total_num, target_bar = 0.0, 0, tqdm(target_loader)
        source_acc_total, target_acc_total = 0, 0
        train_source_iter_labeled=enumerate(source_labeled_loaders)
        train_source_iter_unlabeled = enumerate(source_unlabeled_loaders)
        setup_seed(20)
        for data_target, label_target in target_bar:
            _,(data_source,labels_source) = next(train_source_iter_labeled)
            _, (x_un, _) = next(train_source_iter_unlabeled)
            x_un = x_un.to(device)
            data_source, labels_source = data_source.to(device), labels_source.to(device)
            data_target, labels_target = data_target.to(device), label_target.to(device)
            if parameter['T_DANN']:
                tripleada = 0
            else:
                tripleada = 1
            if epoch >= threshold:
                pred, domain_loss, ajloss, contrastive_loss, sim_weight, L2 = model(torch.cat((data_source, x_un, data_target)), tripleada=tripleada, threshold=1)
            else:
                pred, domain_loss, ajloss, contrastive_loss, sim_weight, L2 = model(torch.cat((data_source, data_target)), tripleada=0, threshold=0)


            source_pred = pred[0:len(data_source), :]
            target_pred = pred[-len(data_source):, :]
            if epoch >= threshold:
                log_prob = torch.nn.functional.log_softmax(sim_weight * source_pred, dim=1)
                # log_prob = torch.nn.functional.log_softmax(source_pred, dim=1)
            else:
                log_prob = torch.nn.functional.log_softmax(source_pred, dim=1)

            celoss = -torch.sum(log_prob * labels_source)/ len(labels_source)
            loss = celoss + parameter['DANN']*domain_loss + parameter['dynamic_adj']*ajloss + parameter['GCL']*contrastive_loss



            source_scores = source_pred.detach().argmax(dim=1)
            source_acc = (source_scores == labels_source.argmax(dim=1)).float().sum().item()
            source_acc_total += source_acc
            target_scores = target_pred.detach().argmax(dim=1)
            target_acc = (target_scores == labels_target.argmax(dim=1)).float().sum().item()
            target_acc_total += target_acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_num += parameter['batch_size']
            total_loss += loss.item() * parameter['batch_size']
            epoch_train_loss = total_loss / total_num
            target_bar.set_description('sub:{} Train Epoch: [{}/{}] Loss: {:.4f} source_acc:{:.2f}% target_acc:{:.2f}%'.format(subject_id, epoch+1, parameter['epochs'], epoch_train_loss, source_acc_total/total_num * 100,
                                                                                                                      target_acc_total/total_num * 100))
        target_test_acc = test_GCN_contrast(model, target_loader, device)
        acc_list[epoch] = target_test_acc
        if best_acc < (target_acc_total / total_num):
            best_acc = (target_acc_total / total_num)

        if best_test_acc < target_test_acc:
            best_test_acc = target_test_acc
            os.chdir(sys_path+'\\model_result')
            torch.save(model.state_dict(), 'model_semi_session_{}_batch48_{}epoch_{}U_sub{}.pkl'.format(Dataset_name, parameter['threshold'], parameter['num_of_U'], subject_id))
    print("best_acc:", best_acc, "best_test_acc:", best_test_acc)
    return acc_list, best_test_acc

def test_GCN_contrast(model, target_loader, device):
    model.eval()
    data_target = target_loader.dataset.tensors[0].to(device)
    labels_target = target_loader.dataset.tensors[1].to(device)
    pred = model.predict(data_target)
    target_scores = pred.detach().argmax(dim=1)
    target_acc = ((target_scores == labels_target.argmax(dim=1)).float().sum().item())/len(data_target)
    print("target_acc:",target_acc)
    return target_acc

def get_dataset_V(test_id,parameter):
    path=[]
    path.append(sys_path+'\\SEED-V_DE\\feature_for_net_session1_de_V')
    path.append(sys_path+'\\SEED-V_DE\\feature_for_net_session2_de_V')
    path.append(sys_path+'\\SEED-V_DE\\feature_for_net_session3_de_V')
    video_length_info = sys_path + '\\de_feature\\video_length.mat'
    video_time = scio.loadmat(video_length_info)['video_length']
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    feature_list_source_labeled=[]
    label_list_source_labeled=[]
    feature_list_source_unlabeled=[]
    label_list_source_unlabeled=[]
    feature_list_target=[]
    label_list_target=[]
    for session in range(3):
        index=0
        u_list = []
        for u in range(parameter["num_of_U"]):
            if test_id + u + 1 >= 16:
                u_list.append(test_id + u + 1 - 16)
            else:
                u_list.append(test_id + u + 1)
        for i in range(16):
            info = os.listdir(path[session])
            domain = os.path.abspath(path[session])
            info = os.path.join(domain, info[i])  # 将路径与文件名结合起来就是每个文件的完整路径
            feature = scio.loadmat(info)['dataset']['feature'][0, 0]
            label = scio.loadmat(info)['dataset']['label'][0, 0]
            one_hot_label_mat = np.zeros((len(label), 5))
            for i in range(len(label)):
                if label[i] == 0:
                    one_hot_label = [1, 0, 0, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 1:
                    one_hot_label = [0, 1, 0, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 2:
                    one_hot_label = [0, 0, 1, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 3:
                    one_hot_label = [0, 0, 0, 1, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 4:
                    one_hot_label = [0, 0, 0, 0, 1]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 5)
                    one_hot_label_mat[i, :] = one_hot_label

            if index != test_id:
                feature = min_max_scaler.fit_transform(feature).astype('float32')
                feature = feature.reshape(feature.shape[0], 62, 5, order='F')
                if parameter['semi'] == 1:
                    ## source unlabeled data
                    if index in u_list:
                        feature_unlabeled = feature
                        label_unlabeled = one_hot_label_mat
                        feature_list_source_unlabeled.append(feature_unlabeled)
                        label_list_source_unlabeled.append(label_unlabeled)
                    else:
                        ## source labeled data
                        feature_labeled = feature
                        label_labeled = one_hot_label_mat
                        feature_list_source_labeled.append(feature_labeled)
                        label_list_source_labeled.append(label_labeled)
                elif parameter['semi'] == 2:
                    video = 4
                    feature_labeled = feature[0:np.cumsum(video_time[0:video])[-1], :]
                    label_labeled = one_hot_label_mat[0:np.cumsum(video_time[0:video])[-1], :]
                    feature_unlabeled = feature[np.cumsum(video_time[0:video])[-1]:len(feature), :]
                    label_unlabeled = one_hot_label_mat[np.cumsum(video_time[0:video])[-1]:len(feature), :]

                    feature_list_source_labeled.append(feature_labeled)
                    label_list_source_labeled.append(label_labeled)
                    feature_list_source_unlabeled.append(feature_unlabeled)
                    label_list_source_unlabeled.append(label_unlabeled)
                else:
                    feature_labeled = feature
                    label_labeled = one_hot_label_mat
                    feature_list_source_labeled.append(feature_labeled)
                    label_list_source_labeled.append(label_labeled)
            else:
                feature = min_max_scaler.fit_transform(feature).astype('float32')
                feature = feature.reshape(feature.shape[0], 62, 5, order='F')
                feature_list_target.append(feature)
                label_list_target.append(one_hot_label_mat)
            index += 1
    source_feature_labeled,source_label_labeled=np.vstack(feature_list_source_labeled),np.vstack(label_list_source_labeled)
    source_feature_unlabeled,source_label_unlabeled=np.vstack(feature_list_source_unlabeled),np.vstack(label_list_source_unlabeled)
    target_feature=np.vstack(feature_list_target)
    target_label=np.vstack(label_list_target)
    target_set={'feature':target_feature,'label':target_label}
    source_set_labeled={'feature':source_feature_labeled,'label':source_label_labeled}
    source_set_unlabeled={'feature':source_feature_unlabeled,'label':source_label_unlabeled}
    return target_set,source_set_labeled,source_set_unlabeled

def get_dataset_IV(test_id, parameter):
    path=[]
    path.append(sys_path+'\\de_feature\\feature_for_net_session'+str(1)+'_LDS_de_IV')
    path.append(sys_path+'\\de_feature\\feature_for_net_session'+str(2)+'_LDS_de_IV')
    path.append(sys_path+'\\de_feature\\feature_for_net_session'+str(3)+'_LDS_de_IV')
    ## our label:0 negative, label:1 :neural,label:2:positive, seed original label: -1,0,1, our label= seed label+1
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    video_length_info = sys_path+'\\de_feature\\video_length.mat'
    video_time=scio.loadmat(video_length_info)['video_length']
    feature_list_source_labeled=[]
    label_list_source_labeled=[]
    feature_list_source_unlabeled=[]
    label_list_source_unlabeled=[]
    feature_list_target=[]
    label_list_target=[]

    for session in range(3):
        index=0
        u_list = []
        for u in range(parameter["num_of_U"]):
            if test_id + u + 1 >= 15:
                u_list.append(test_id + u + 1 - 15)
            else:
                u_list.append(test_id + u + 1)
        for i in range(15):
            info = os.listdir(path[session])
            domain = os.path.abspath(path[session])
            info = os.path.join(domain,info[i]) #将路径与文件名结合起来就是每个文件的完整路径
            feature = scio.loadmat(info)['dataset']['feature'][0,0]
            label = scio.loadmat(info)['dataset']['label'][0,0]
            one_hot_label_mat = np.zeros((len(label), 4))
            for i in range(len(label)):
                if label[i] == 0:
                    one_hot_label = [1, 0, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 1:
                    one_hot_label = [0, 1, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 2:
                    one_hot_label = [0, 0, 1, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
                    one_hot_label_mat[i, :] = one_hot_label
                if label[i] == 3:
                    one_hot_label = [0, 0, 0, 1]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
                    one_hot_label_mat[i, :] = one_hot_label

            if index!=test_id:
                feature = min_max_scaler.fit_transform(feature).astype('float32')
                feature = feature.reshape(feature.shape[0], 62, 5, order='F')
                if parameter['semi'] == 1:
                    ## source unlabeled data
                    if index in u_list:
                        feature_unlabeled = feature
                        label_unlabeled = one_hot_label_mat
                        feature_list_source_unlabeled.append(feature_unlabeled)
                        label_list_source_unlabeled.append(label_unlabeled)
                    else:
                        ## source labeled data
                        feature_labeled = feature
                        label_labeled = one_hot_label_mat
                        feature_list_source_labeled.append(feature_labeled)
                        label_list_source_labeled.append(label_labeled)
                elif parameter['semi'] == 2:
                    video = 4
                    feature_labeled = feature[0:np.cumsum(video_time[0:video])[-1], :]
                    label_labeled = one_hot_label_mat[0:np.cumsum(video_time[0:video])[-1], :]
                    feature_unlabeled = feature[np.cumsum(video_time[0:video])[-1]:len(feature), :]
                    label_unlabeled = one_hot_label_mat[np.cumsum(video_time[0:video])[-1]:len(feature), :]

                    feature_list_source_labeled.append(feature_labeled)
                    label_list_source_labeled.append(label_labeled)
                    feature_list_source_unlabeled.append(feature_unlabeled)
                    label_list_source_unlabeled.append(label_unlabeled)
                else:
                    feature_labeled = feature
                    label_labeled = one_hot_label_mat
                    feature_list_source_labeled.append(feature_labeled)
                    label_list_source_labeled.append(label_labeled)
            else:
                feature = min_max_scaler.fit_transform(feature).astype('float32')
                feature = feature.reshape(feature.shape[0], 62, 5, order='F')
                feature_list_target.append(feature)
                label_list_target.append(one_hot_label_mat)
            index+=1

    source_feature_labeled,source_label_labeled=np.vstack(feature_list_source_labeled),np.vstack(label_list_source_labeled)
    source_feature_unlabeled,source_label_unlabeled=np.vstack(feature_list_source_unlabeled),np.vstack(label_list_source_unlabeled)
    target_feature=np.vstack(feature_list_target)
    target_label=np.vstack(label_list_target)

    target_set={'feature':target_feature,'label':target_label}
    source_set_labeled={'feature':source_feature_labeled,'label':source_label_labeled}
    source_set_unlabeled={'feature':source_feature_unlabeled,'label':source_label_unlabeled}

    return target_set,source_set_labeled,source_set_unlabeled

def get_dataset(test_id, session, parameter):
    session =session+1
    path = sys_path+'\\de_feature\\feature_for_net_session' + str(session) + '_LDS_de'
    os.chdir(path)
    feature_list_source_labeled = []
    label_list_source_labeled = []
    feature_list_source_unlabeled = []
    label_list_source_unlabeled = []
    feature_list_target = []
    label_list_target = []
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    video_time = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
    index = 0
    u_list = []
    for u in range(parameter["num_of_U"]):
        if test_id+u+1>=15:
            u_list.append(test_id+u+1-15)
        else:
            u_list.append(test_id+u+1)
    for info in os.listdir(path):
        domain = os.path.abspath(path)
        info_ = os.path.join(domain, info)  # 将路径与文件名结合起来就是每个文件的完整路径
        if session == 1:
            feature = scio.loadmat(info_)['dataset_session1']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session1']['label'][0, 0]
        elif session == 2:
            feature = scio.loadmat(info_)['dataset_session2']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session2']['label'][0, 0]
        else:
            feature = scio.loadmat(info_)['dataset_session3']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session3']['label'][0, 0]


        one_hot_label_mat = np.zeros((len(label), 3))
        for i in range(len(label)):
            if label[i] == 0:
                one_hot_label = [1, 0, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            if label[i] == 1:
                one_hot_label = [0, 1, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            if label[i] == 2:
                one_hot_label = [0, 0, 1]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label


        if index != test_id:
            feature = min_max_scaler.fit_transform(feature).astype('float32')
            feature = feature.reshape(feature.shape[0], 62, 5, order='F')
            if parameter['semi']==1:
                ## source unlabeled data
                if index in u_list:
                    feature_unlabeled = feature
                    label_unlabeled = one_hot_label_mat
                    feature_list_source_unlabeled.append(feature_unlabeled)
                    label_list_source_unlabeled.append(label_unlabeled)
                else:
                    ## source labeled data
                    feature_labeled = feature
                    label_labeled = one_hot_label_mat
                    feature_list_source_labeled.append(feature_labeled)
                    label_list_source_labeled.append(label_labeled)
            elif parameter['semi']==2:
                video=3
                feature_labeled = feature[0:np.cumsum(video_time[0:video])[-1], :]
                label_labeled = one_hot_label_mat[0:np.cumsum(video_time[0:video])[-1], :]
                feature_unlabeled = feature[np.cumsum(video_time[0:video])[-1]:len(feature), :]
                label_unlabeled = one_hot_label_mat[np.cumsum(video_time[0:video])[-1]:len(feature), :]

                feature_list_source_labeled.append(feature_labeled)
                label_list_source_labeled.append(label_labeled)
                feature_list_source_unlabeled.append(feature_unlabeled)
                label_list_source_unlabeled.append(label_unlabeled)
            else:
                feature_labeled = feature
                label_labeled = one_hot_label_mat
                feature_list_source_labeled.append(feature_labeled)
                label_list_source_labeled.append(label_labeled)
        else:
            feature = min_max_scaler.fit_transform(feature).astype('float32')
            feature = feature.reshape(feature.shape[0], 62, 5, order='F')
            ## target unlabeled data
            feature_list_target.append(feature)
            label_list_target.append(one_hot_label_mat)

        index += 1

    source_feature_labeled, source_label_labeled = np.vstack(feature_list_source_labeled), np.vstack(label_list_source_labeled)
    source_feature_unlabeled, source_label_unlabeled = np.vstack(feature_list_source_unlabeled), np.vstack(label_list_source_unlabeled)
    target_feature = feature_list_target[0]
    target_label = label_list_target[0]

    target_set = {'feature': target_feature, 'label': target_label}
    source_set_labeled = {'feature': source_feature_labeled, 'label': source_label_labeled}
    source_set_unlabeled = {'feature': source_feature_unlabeled, 'label': source_label_unlabeled}

    return target_set, source_set_labeled, source_set_unlabeled

def cross_subject(target_set, source_set_labeled, source_set_unlabeled, subject_id, parameter, net_params):
    setup_seed(20)
    torch_dataset_test = Data.TensorDataset(torch.from_numpy(target_set['feature']), torch.from_numpy(target_set['label']))
    torch_dataset_source_labeled = Data.TensorDataset(torch.from_numpy(source_set_labeled['feature']), torch.from_numpy(source_set_labeled['label']))
    torch_dataset_source_unlabeled = Data.TensorDataset(torch.from_numpy(source_set_unlabeled['feature']), torch.from_numpy(source_set_unlabeled['label']))

    source_labeled_loaders = torch.utils.data.DataLoader(dataset=torch_dataset_source_labeled,
                                                 batch_size=parameter['batch_size'],
                                                 shuffle=True,
                                                 drop_last=True)

    source_unlabeled_loaders = torch.utils.data.DataLoader(dataset=torch_dataset_source_unlabeled,
                                                 batch_size=parameter['batch_size'],
                                                 shuffle=True,
                                                 drop_last=True)

    target_loader = torch.utils.data.DataLoader(dataset=torch_dataset_test,
                                                batch_size=parameter['batch_size'],
                                                shuffle=True,
                                                drop_last=True)

    acc = train_GCN_contrast(subject_id, parameter, net_params, source_labeled_loaders, source_unlabeled_loaders, target_loader)
    return acc

def main(parameter, net_params):
   # data preparation
    setup_seed(20)
    if net_params["category_number"] == 3:
        print('Model name: MS-MDAER. Dataset name: SEED')
        sub_num = 15
    elif net_params["category_number"] == 4:
        print('Model name: MS-MDAER. Dataset name: SEED_IV')
        sub_num = 15
    else:
        print('Model name: MS-MDAER. Dataset name: SEED_V')
        sub_num = 16
    print('BS: {}, epoch: {}'.format(parameter['batch_size'], parameter['epochs']))
    # store the results
    csub = []
    # for session_id_main in range(3):
    session_id = 0
    # subject_id = 0

    best_acc_mat = np.zeros(sub_num)
    target_acc_curve = np.zeros((sub_num, parameter['epochs']))


    global Dataset_name
    for subject_id in range(sub_num):
        if net_params['category_number'] == 3:
            target_set, source_set_labeled, source_set_unlabeled = get_dataset(subject_id, session_id, parameter)
            Dataset_name = 'SEED'
        elif net_params['category_number'] == 4:
            target_set, source_set_labeled, source_set_unlabeled = get_dataset_IV(subject_id, parameter)
            Dataset_name = 'SEEDIV'
        elif net_params['category_number'] == 5:
            target_set, source_set_labeled, source_set_unlabeled = get_dataset_V(subject_id, parameter)
            Dataset_name = 'SEEDV'
        else:
            pass
        acc = cross_subject(target_set, source_set_labeled, source_set_unlabeled, subject_id, parameter, net_params)
        csub.append(acc[1])
        target_acc_curve[subject_id, :] = acc[0]
        best_acc_mat[subject_id] = acc[1]
    print("Cross-subject: ", csub)

    result_list = {'best_acc_mat': best_acc_mat,
               'target_acc_curve': target_acc_curve}
    os.chdir(sys_path+'\\model_result')
    np.save('result_list_semi_session_{}_batch48_{}epoch_{}U.npy'.format(Dataset_name, parameter['threshold'], parameter['num_of_U']), result_list)
    return csub

parameter = {'epochs':100, 'batch_size':48, 'init_lr':1e-3, 'weight_decay':1e-5, "semi": 1, 'threshold': 30, 'num_of_U':2, 'drop_rate':0.8,
             'GLalpha': 0.01, 'node_feature_hidden1': 5,'linearsize': 128, 'category_number': 5, 'DEVICE': 'cuda:0', 'K': 3, 'num_of_vertices': 62, 'num_of_features': 5,
             'GCL': 1, 'dynamic_adj': 1, 'DANN': 1, 'Multi_att': 1, 'T_DANN': 1}

csub = main(parameter, parameter)

c=np.load('result_list_semi_session_{}_batch48_{}epoch_{}U.npy'.format(Dataset_name, parameter['threshold'], parameter['num_of_U']), allow_pickle=True).item()
c_mean=np.mean(c['best_acc_mat'])
c_std=np.std(c['best_acc_mat'])




