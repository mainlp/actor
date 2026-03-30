import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch
import torch.nn as nn
import gc


class DALMLP(nn.Module):
    def __init__(self, emb_dim: int, num_labels: int, dropout: float = 0.1):
        super(DALMLP, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.fc3 = nn.Linear(emb_dim, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class DALEstimator:
    def __init__(self, args, emb_dim, num_labels, class_weights, device):
        self.model = DALMLP(emb_dim, num_labels)
        self.device = device
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.dal_learning_rate, weight_decay=args.dal_weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.class_weights = class_weights
    
    def train(self, epochs, train_dataloader, args):
        self.model.train()
        for epoch in range(epochs):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                x, y = batch 
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        return self.model
    
    def predict(self, x):
        self.model.eval()
        x = x.to(self.device)
        logits = self.model(x)
        return logits





class AcquisitionMethod:
    def __init__(self, args, model, num_labels, device, unlabeled_train_dataset, unlabeled_idxs):
        self.args = args
        self.model = model
        self.num_labels = num_labels
        self.device = device
        self.unlabeled_train_dataset = unlabeled_train_dataset
        self.unlabeled_idxs = unlabeled_idxs

    def get_logits_maj(self, args):
        """
        get the logits of the model on the training set
        :param X_train: the training set
        :param Y_train: the training labels
        :param labeled_idx: the indices of the labeled examples
        :return: the logits of the model on the training set
        """
        unlabeled_train_dataloader = DataLoader(self.unlabeled_train_dataset, batch_size=args.eval_batch_size, shuffle=False)
        self.model.eval()
        logits_list = []
        text_ids_list = []
        for step, batch in enumerate(tqdm(unlabeled_train_dataloader, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            if args.sampling_strategy is not None:
                input_ids, input_mask, segment_ids, label_ids, text_ids = batch
                text_ids_list.append(text_ids.detach().cpu())
            else:
                input_ids, input_mask, segment_ids, label_ids = batch
            logits = self.model(input_ids, segment_ids, input_mask, labels=None)
            logits_list.append(logits.detach().cpu())
        logits_list =  torch.cat(logits_list, dim=0)
        text_ids_list = np.concatenate(text_ids_list)
        return logits_list, text_ids_list

        
    def query_maj(self,args):
        """
        get the indices of labeled examples after the given amount have been queried by the query strategy.
        :param X_train: the training set
        :param Y_train: the training labels
        :param labeled_idx: the indices of the labeled examples
        :param amount: the amount of examples to query
        :return: the new labeled indices (including the ones queried)
        """
        
        return NotImplemented
    

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model



class Random(AcquisitionMethod):
    def __init__(self, args, model, num_labels, device, unlabeled_train_dataset, unlabeled_idxs):
        super(Random, self).__init__(args, model, num_labels, device, unlabeled_train_dataset, unlabeled_idxs)
    
    def query_maj(self, args):
        return np.random.choice(self.unlabeled_idxs, args.query_sample_size, replace=False)

class EntropyMajority(AcquisitionMethod):
    def __init__(self, args, model, num_labels, device, unlabeled_train_dataset, unlabeled_idxs):
        super(EntropyMajority, self).__init__(args, model, num_labels, device,  unlabeled_train_dataset, unlabeled_idxs)

    def query_maj(self, args):
        logits_list, text_ids = self.get_logits_maj(args) # get logits of the model on the training set
        probs = F.softmax(logits_list, dim=1) # calculate softmax values
        log_probs = F.log_softmax(logits_list, dim=1) # calculate log of softmax values
        entropy = -(probs * log_probs).sum(1).numpy() # calculate entropy
        if args.sampling_strategy == 'label_first':
            topk_idx = np.argsort(entropy, )[-args.query_sample_size:] # select the last k samples with highest entropy
        elif args.sampling_strategy == 'instance_first':

            sorted_idx = np.argsort(entropy, )[::-1]
            # return the top k idx, but skip if the text id is already in the selected list
            topk_idx = []
            seletect_text_ids = []
            for idx in sorted_idx:
                if text_ids[idx] not in seletect_text_ids:
                    topk_idx.append(idx)
                    seletect_text_ids.append(text_ids[idx])
                if len(topk_idx) == args.query_sample_size:
                    break
        return self.unlabeled_idxs[topk_idx] # return the indices of the selected samples




class LeastConfidence(AcquisitionMethod):
    def __init__(self, args, model, num_labels, device, unlabeled_train_dataset, unlabeled_idxs):
        super(LeastConfidence, self).__init__(args, model, num_labels, device, unlabeled_train_dataset, unlabeled_idxs)

    def query_maj(self, args):
        logits_list, text_ids = self.get_logits_maj(args) # get logits of the model on the training set
        probs = F.softmax(logits_list, dim=1) # calculate softmax values
        confidence = probs.max(1)[0].numpy() # calculate confidence
        # select the  k samples with lowest confidence
        topk_idx = np.argsort(confidence)[:args.query_sample_size]
        if args.sampling_strategy == 'label_first':
            topk_idx = np.argsort(entropy, )[-args.query_sample_size:] # select the last k samples with highest entropy
        elif args.sampling_strategy == 'instance_first':
            topk_idx = np.argsort(entropy, )[:args.query_sample_size]
        return self.unlabeled_idxs[topk_idx] # return the indices of the selected samples


class DAL(AcquisitionMethod):
    def __init__(self, args, model, num_labels, device, estimator,  unlabeled_train_dataset, unlabeled_idxs, labeled_idxs, labeled_train_dataset):
        super(DAL, self).__init__(args, model, num_labels, device , unlabeled_train_dataset, unlabeled_idxs)
        self.estimator = estimator
        self.labeled_idxs = labeled_idxs
        self.unlabeled_idxs = unlabeled_idxs
        self.unlabeled_train_dataset = unlabeled_train_dataset
        self.labeled_train_dataset = labeled_train_dataset
    
    def get_representation(self, args, dataset):
        dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)
        self.model.eval()
        representation_list = []
        text_ids_list = []
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, text_ids = batch
            text_ids_list.append(text_ids.detach().cpu())
            # get the representation of the [cls] token from last layer for each example in the dataset

            representation = self.model.bert(input_ids, segment_ids, input_mask)[1]
            representation_list.append(representation.detach().cpu())

        text_ids_list = np.concatenate(text_ids_list)
        representation_list =  torch.cat(representation_list, dim=0)
        return representation_list, text_ids_list

    def estimator_train(self, args,  labeled_representation, unlabeled_representation):
        
        # constructing the training set for the estimator by combining the labeled and unlabeled representation
        #  labels are 0 for labeled examples and 1 (int) for unlabeled examples  
        train_representation = torch.cat([labeled_representation, unlabeled_representation], dim=0)


        train_labels = torch.cat([torch.zeros(labeled_representation.shape[0], dtype=torch.int64), torch.ones(unlabeled_representation.shape[0], dtype=torch.int64)], dim=0)
        train_dataset = TensorDataset(train_representation, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        self.estimator.train(epochs=args.dal_epochs, train_dataloader=train_dataloader, args=args)
        
        return self.estimator




    def query_maj(self, args):
        # get the representation of the labeled and unlabeled data
        labeled_representation, text_ids = self.get_representation(args, self.labeled_train_dataset)
        unlabeled_representation, text_ids = self.get_representation(args, self.unlabeled_train_dataset)
        # train the estimator using the labeled and unlabeled representation
        self.estimator_train(args, labeled_representation, unlabeled_representation)


        # get the logits of the estimator on the unlabeled data
        logits_list = self.estimator.predict(unlabeled_representation)
        probs = F.softmax(logits_list, dim=1).detach().cpu() # calculate softmax values
        probs = probs[:, 1]
        # sort the probs based on the probability of the unlabeled example
        # topk_idx = np.argpartition(probs[:, 1],  -args.query_sample_size)[-args.query_sample_size:]
        if args.sampling_strategy == 'label_first':
            topk_idx = np.argsort(probs, )[-args.query_sample_size:] # select the last k samples with highest entropy
        elif args.sampling_strategy == 'instance_first':
            sorted_idx = torch.argsort(probs, descending=True)
            # return the top k idx, but skip if the text id is already in the selected list
            topk_idx = []
            seletect_text_ids = []
            for idx in sorted_idx:
                if text_ids[idx] not in seletect_text_ids:
                    topk_idx.append(idx)
                    seletect_text_ids.append(text_ids[idx])
                if len(topk_idx) == args.query_sample_size:
                    break
        # # select the  k samples with lowest confidence
        # topk_idx = np.argsort(confidence)[:args.query_sample_size]
        return self.unlabeled_idxs[topk_idx] # return the indices of the selected samples



class BALD(AcquisitionMethod):
    def __init__(self, args, model, num_labels, device, unlabeled_train_dataset, unlabeled_idxs):
        super(BALD, self).__init__(args, model, num_labels, device, unlabeled_train_dataset, unlabeled_idxs)
    
    def enable_dropout(self, model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def dropout_predict(self, data, T):
        self.model.eval()
        self.enable_dropout(self.model)
        predictions = []
        for i in range(T):
            input_ids, input_mask, segment_ids, label_ids, text_ids = data
            logits = self.model(input_ids, segment_ids, input_mask, labels=None)
            predictions.append(F.softmax(logits, dim=1).detach().cpu())
        predictions = torch.stack(predictions)
        return predictions.mean(0), predictions.var(0).sum(1).unsqueeze(1)

    def query_maj(self, args):#
        unlabeled_train_dataloader = DataLoader(self.unlabeled_train_dataset, batch_size=args.eval_batch_size, shuffle=False)
        pred_list = []
        
        text_ids_list = []
        for step, batch in enumerate(tqdm(unlabeled_train_dataloader, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, text_ids = batch
            text_ids_list.append(text_ids.detach().cpu())
            pred, _ = self.dropout_predict(batch, args.T)
            pred_list.append(pred.detach().cpu())
        pred_list = torch.cat(pred_list, dim=0)
        text_ids = np.concatenate(text_ids_list)
        
        entropy = -torch.sum(pred_list * torch.log(pred_list), dim=1)
        if args.sampling_strategy == 'label_first':
            topk_idx = np.argsort(entropy, )[-args.query_sample_size:] # select the last k samples with highest entropy
        elif args.sampling_strategy == 'instance_first':
            sorted_idx = torch.argsort(entropy, descending=True)
            # return the top k idx, but skip if the text id is already in the selected list
            topk_idx = []
            seletect_text_ids = []
            for idx in sorted_idx:
                if text_ids[idx] not in seletect_text_ids:
                    topk_idx.append(idx)
                    seletect_text_ids.append(text_ids[idx])
                if len(topk_idx) == args.query_sample_size:
                    break

        return self.unlabeled_idxs[topk_idx] # return the indices of the selected samples
        
        


def calculate_entropy_select_anno_list_group_level_askall(args, model,unlabeled_idxs, unlabeled_train_dataset, device):
    k, strategy = args.query_sample_size, args.active_strategy
    # calculate entropy of unlabeled data
    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, batch_size=args.train_batch_size, shuffle=False)
    model.eval()
    entropy = []
    for step, batch in enumerate(tqdm(unlabeled_train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, annotator_ids = batch
        loss, logits, logits_all= model(input_ids, segment_ids, input_mask, labels=None, annotator_ids=annotator_ids, pred_mode=args.eval_mode, pred=True)
        logits = sum(logits_all)
        probs = F.softmax(logits)
        log_probs = F.log_softmax(logits)
        entropy.append(-(probs * log_probs).sum(1).detach().cpu().numpy())
    entropy = np.concatenate(entropy)

    topk_idx = np.argsort(entropy, )[-k:]
    return unlabeled_idxs[topk_idx]




def calculate_entropy_select_anno_group_level(args, model,unlabeled_idxs, unlabeled_train_dataset, device,):
    # calculate entropy of unlabeled data
    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, batch_size=args.train_batch_size, shuffle=False)
    model.eval()
    entropy = []
    text_ids_list = []
    for step, batch in enumerate(tqdm(unlabeled_train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, annotator_ids , text_ids= batch
        loss, logits, logits_all, pred_all= model(input_ids, segment_ids, input_mask, labels=None, annotator_ids=annotator_ids, pred_mode='majority', pred=True)
        logits = sum(logits_all)
        probs = F.softmax(logits)
        log_probs = F.log_softmax(logits)
        entropy.append(-(probs * log_probs).sum(1).detach().cpu().numpy())
        text_ids_list.append(text_ids.detach().cpu().numpy())

    entropy = np.concatenate(entropy)
    text_ids = np.concatenate(text_ids_list)
    seletect_text_ids = []
    if args.sampling_strategy == 'label_first':
        topk_idx = np.argsort(entropy, )[-args.query_sample_size:] # select the last k samples with highest entropy
    elif args.sampling_strategy == 'instance_first':
        sorted_idx = np.argsort(entropy, )[::-1]
        # return the top k idx, but skip if the text id is already in the selected list
        topk_idx = []
        
        for idx in sorted_idx:
            if text_ids[idx] not in seletect_text_ids:
                topk_idx.append(idx)
                seletect_text_ids.append(text_ids[idx])
            if len(topk_idx) == args.query_sample_size:
                break
    return unlabeled_idxs[topk_idx]



def calculate_entropy_norm_select_anno_group_level(args, model,unlabeled_idxs, unlabeled_train_dataset, device):
    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, batch_size=args.train_batch_size, shuffle=False)
    model.eval()
    entropy = []
    text_ids_list = []
    for step, batch in enumerate(tqdm(unlabeled_train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, annotator_ids , text_ids= batch
        loss, logits, logits_all, pred_all= model(input_ids, segment_ids, input_mask, labels=None, annotator_ids=annotator_ids, pred_mode='majority', pred=True)
        # normalize the logits
        logits_norm = []
        for logit in logits_all:
            logit_norm = F.normalize(logit, p=2, dim=1)
            logits_norm.append(logit_norm)
        logits = sum(logits_norm)
        probs = F.softmax(logits)
        log_probs = F.log_softmax(logits)
        entropy.append(-(probs * log_probs).sum(1).detach().cpu().numpy())
        text_ids_list.append(text_ids.detach().cpu().numpy())

    entropy = np.concatenate(entropy)
    text_ids = np.concatenate(text_ids_list)
    seletect_text_ids = []
    if args.sampling_strategy == 'label_first':
        topk_idx = np.argsort(entropy, )[-args.query_sample_size:] # select the last k samples with highest entropy
    elif args.sampling_strategy == 'instance_first':
        sorted_idx = np.argsort(entropy, )[::-1]
        # return the top k idx, but skip if the text id is already in the selected list
        topk_idx = []
        
        for idx in sorted_idx:
            if text_ids[idx] not in seletect_text_ids:
                topk_idx.append(idx)
                seletect_text_ids.append(text_ids[idx])
            if len(topk_idx) == args.query_sample_size:
                break
    return unlabeled_idxs[topk_idx]





def calculate_entropy_select_anno_indi_level(args, model,unlabeled_idxs, unlabeled_train_dataset, device):
    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, batch_size=args.train_batch_size, shuffle=False)
    model.eval()
    entropy = []
    text_ids_list = []
    for step, batch in enumerate(tqdm(unlabeled_train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, annotator_ids, text_ids = batch
        loss, logits, logits_all, pred_all= model(input_ids, segment_ids, input_mask, labels=None, annotator_ids=annotator_ids, pred_mode='majority', pred=True)
        # for each sample in the batch, only consider the logits from the annotator head , which can be found by indexing with annotator_ids
        # logits_all is a list of logits from all annotators, annotator_ids is a list of annotator ids for each sample in the batch
        text_ids_list.append(text_ids.detach().cpu().numpy())
        for i in range(len(annotator_ids)):
            logits = logits_all[annotator_ids[i]][i]
            probs = F.softmax(logits)
            log_probs = F.log_softmax(logits)
            entropy.append(-(probs * log_probs).sum().detach().cpu().numpy())
    # if entropy list is empty, stop program
    if len(entropy) == 0:
        print('entropy list is empty, stop program')
        exit()
    entropy = np.array(entropy)
    text_ids = np.concatenate(text_ids_list)

    if args.sampling_strategy == 'label_first':
        topk_idx = np.argsort(entropy, )[-args.query_sample_size:] # select the last k samples with highest entropy
    elif args.sampling_strategy == 'instance_first':
        sorted_idx = np.argsort(entropy, )[::-1]
        # return the top k idx, but skip if the text id is already in the selected list
        topk_idx = []
        seletect_text_ids = []
        for idx in sorted_idx:
            if text_ids[idx] not in seletect_text_ids:
                topk_idx.append(idx)
                seletect_text_ids.append(text_ids[idx])
            if len(topk_idx) == args.query_sample_size:
                break
    return unlabeled_idxs[topk_idx]

def calculate_entropy_select_anno_indi_and_group_level(args, model,unlabeled_idxs, unlabeled_train_dataset, device):
    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, batch_size=args.train_batch_size, shuffle=False)
    model.eval()
    entropy_group = []
    entropy_indi = []
    text_ids_list = []
    for step, batch in enumerate(tqdm(unlabeled_train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, annotator_ids , text_ids= batch
        loss, logits, logits_all, pred_all= model(input_ids, segment_ids, input_mask, labels=None, annotator_ids=annotator_ids, pred_mode='majority', pred=True)
        logits = sum(logits_all)
        probs = F.softmax(logits)
        log_probs = F.log_softmax(logits)
        entropy_group.append(-(probs * log_probs).sum(1).detach().cpu().numpy())
        text_ids_list.append(text_ids.detach().cpu().numpy())
        for i in range(len(annotator_ids)):
            logits = logits_all[annotator_ids[i]][i]
            probs = F.softmax(logits)
            log_probs = F.log_softmax(logits)
            entropy_indi.append(-(probs * log_probs).sum().detach().cpu().numpy())
    if len(entropy_indi) == 0:
        print('entropy list is empty, stop program')
        exit()
    entropy_group = np.concatenate(entropy_group)
    entropy_indi = np.array(entropy_indi)
    entropy = entropy_group + entropy_indi
    text_ids = np.concatenate(text_ids_list)
    seletect_text_ids = []
    if args.sampling_strategy == 'label_first':
        topk_idx = np.argsort(entropy, )[-args.query_sample_size:] # select the last k samples with highest entropy
    elif args.sampling_strategy == 'instance_first':
        sorted_idx = np.argsort(entropy, )[::-1]
        # return the top k idx, but skip if the text id is already in the selected list
        topk_idx = []
        seletect_text_ids = []
        for idx in sorted_idx:
            if text_ids[idx] not in seletect_text_ids:
                topk_idx.append(idx)
                seletect_text_ids.append(text_ids[idx])
            if len(topk_idx) == args.query_sample_size:
                break
    return unlabeled_idxs[topk_idx]





def calculate_vote_var_select_anno_group_level(args, model,unlabeled_idxs, unlabeled_train_dataset, device):
    # calculate entropy of unlabeled data
    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, batch_size=args.train_batch_size, shuffle=False)
    model.eval()
    entropy = []
    text_ids_list = []
    pred_heads = []
    for step, batch in enumerate(tqdm(unlabeled_train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, annotator_ids , text_ids= batch
        loss, logits, logits_all, pred_all= model(input_ids, segment_ids, input_mask, labels=None, annotator_ids=annotator_ids, pred_mode='majority', pred=True)
        pred_heads.append(pred_all)
        text_ids_list.append(text_ids.detach().cpu().numpy())

    # entropy = np.concatenate(entropy)
    text_ids = np.concatenate(text_ids_list)
    pred_heads = np.concatenate(pred_heads, axis=1)

    var_all = np.var(pred_heads, axis=0)
    seletect_text_ids = []
    if args.sampling_strategy == 'label_first':
        topk_idx = np.argsort(var_all, )[-args.query_sample_size:] # select the last k samples with highest variance
    elif args.sampling_strategy == 'instance_first':
        sorted_idx = np.argsort(var_all, )[::-1]
        # return the top k idx, but skip if the text id is already in the selected list
        topk_idx = []
        
        for idx in sorted_idx:
            if text_ids[idx] not in seletect_text_ids:
                topk_idx.append(idx)
                seletect_text_ids.append(text_ids[idx])
            if len(topk_idx) == args.query_sample_size:
                break
    return unlabeled_idxs[topk_idx]


def calculate_random_select_anno_group_level(args, model, unlabeled_idxs, unlabeled_train_dataset, device):
    return np.random.choice(unlabeled_idxs, args.query_sample_size, replace=False)