from acquisitions import * 
from utils.utils import get_class_weight

def apply_acquisition_function(args, model, unlabeled_idxs, unlabeled_train_dataset, device, labeled_train_dataset, labeled_idxs):
    if args.task_name == 'gab_maj':
        if args.active_strategy == 'entropy':
            acquisit_method = EntropyMajority(args, model, args.num_labels, device, unlabeled_train_dataset, unlabeled_idxs)

        elif args.active_strategy == 'random':
            acquisit_method = Random(args, model, args.num_labels, device, unlabeled_train_dataset, unlabeled_idxs)
        elif args.active_strategy == 'least_confidence':
            acquisit_method = LeastConfidence(args, model,args.num_labels, device, unlabeled_train_dataset, unlabeled_idxs)
        elif args.active_strategy == 'dal':
            class_weight = get_class_weight(labeled_train_dataset, unlabeled_train_dataset)
            estimator = DALEstimator(args, 768, args.num_labels, class_weight, device)
            acquisit_method = DAL(args, model,args.num_labels, device, estimator, unlabeled_train_dataset, unlabeled_idxs, labeled_idxs, labeled_train_dataset) 
        elif args.active_strategy == 'bald':
            acquisit_method = BALD(args, model, args.num_labels, device, unlabeled_train_dataset, unlabeled_idxs)
        acquisit_method.update_model(new_model=model)
        top_k = acquisit_method.query_maj(args)


    elif args.task_name == 'gab_anno':
        if args.active_strategy == 'entropy':
            top_k = calculate_entropy_select_anno_group_level(args, model, unlabeled_idxs, unlabeled_train_dataset, device)
        elif args.active_strategy == 'individual_entropy':
            top_k = calculate_entropy_select_anno_indi_level(args, model, unlabeled_idxs, unlabeled_train_dataset, device)
        elif args.active_strategy == 'mix_entropy':
            top_k = calculate_entropy_select_anno_indi_and_group_level(args, model, unlabeled_idxs, unlabeled_train_dataset, device)
        elif args.active_strategy == 'vote_var':
            top_k = calculate_vote_var_select_anno_group_level(args, model, unlabeled_idxs, unlabeled_train_dataset, device)
        elif args.active_strategy == 'random':
            top_k = calculate_random_select_anno_group_level(args, model, unlabeled_idxs, unlabeled_train_dataset, device)
        elif args.active_strategy == 'entropy_norm':
            top_k = calculate_entropy_norm_select_anno_group_level(args, model, unlabeled_idxs, unlabeled_train_dataset, device)
    elif args.task_name == 'gab_anno_list':
        if args.active_strategy =='group_entropy':
            top_k = calculate_entropy_select_anno_list_group_level_askall(args, model, unlabeled_idxs, unlabeled_train_dataset, device)

    
    return top_k, labeled_train_dataset, unlabeled_train_dataset 