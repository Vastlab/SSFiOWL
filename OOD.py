import torch
import pairwisedistances
from DistributionModels import weibull
from clusteringAlgos import clustering


def distances_analysis(pos_cls_name, features_all_classes, args, gpu, negatives):
    max_positive_distances = find_distances_positives(pos_cls_name, features_all_classes, args, gpu)['distances_positive_median']
    #['distances_positive_max']
    pos_features = features_all_classes[pos_cls_name].to(f"cuda:{gpu}")
    neg_features = []
    total_count = 0
    for i, cls in enumerate(negatives):
        print(f"pos_cls_name {pos_cls_name} cls {cls}")
        neg_features.append(negatives[cls])
    neg_features = torch.cat(neg_features).to(f"cuda:{gpu}")
    distances = pairwisedistances.__dict__[args.distance_metric](pos_features, neg_features)
    min_distances = torch.min(distances,dim=1).values.cpu()
    total_count += distances.nelement()
    # from IPython import embed;embed();
    return





def find_distances_positives(pos_cls_name, features_all_classes, args, gpu):
    features = features_all_classes[pos_cls_name].to(f"cuda:{gpu}")
    distances = pairwisedistances.__dict__[args.distance_metric](features, features)
    distances = distances[~torch.eye(distances.shape[0]).type(torch.BoolTensor)].reshape(distances.shape[0],distances.shape[1]-1)
    # from IPython import embed;embed();
    distances_to_return = {}
    distances_to_return['distances_positive_median'] = torch.median(distances,dim=1).values.cpu()
    distances_to_return['distances_positive_min'] = torch.min(distances,dim=1).values.cpu()
    distances_to_return['distances_positive_max'] = torch.max(distances,dim=1).values.cpu()
    distances_to_return['distances_positive_sum'] = torch.sum(distances).cpu()
    distances_to_return['distances_positive_sum_squared'] = torch.sum(distances**2).cpu()
    distances_to_return['total_count'] = distances.nelement()
    return distances_to_return

def find_distances_negatives(pos_cls_name, features_all_classes, args, gpu):
    pos_features = features_all_classes[pos_cls_name].to(f"cuda:{gpu}")
    neg_features = []
    chunk_size = 200
    min_distances = []
    distances_negative_sum = torch.tensor(0)
    distances_negative_sum_squared = torch.tensor(0)
    total_count = 0
    for i, cls in enumerate(set(list(features_all_classes.keys()))-set([pos_cls_name])):
        neg_features.append(features_all_classes[cls])
        if len(neg_features) == chunk_size:
            neg_features = torch.cat(neg_features).to(f"cuda:{gpu}")
            distances = pairwisedistances.__dict__[args.distance_metric](pos_features, neg_features)
            min_distances.append(torch.min(distances,dim=1).values.cpu())
            distances_negative_sum = distances_negative_sum + torch.sum(distances).cpu()
            distances_negative_sum_squared = distances_negative_sum_squared + torch.sum(distances**2).cpu()
            total_count += distances.nelement()
            neg_features = []
    if len(neg_features) > 0:
        # from IPython import embed;embed();
        neg_features = torch.cat(neg_features).to(f"cuda:{gpu}")
        distances = pairwisedistances.__dict__[args.distance_metric](pos_features, neg_features)
        min_distances.append(torch.min(distances,dim=1).values.cpu())
        distances_negative_sum = distances_negative_sum + torch.sum(distances).cpu()
        distances_negative_sum_squared = distances_negative_sum_squared + torch.sum(distances ** 2).cpu()
        total_count += distances.nelement()
    # from IPython import embed;embed();
    min_distances = torch.stack(min_distances,dim = 1)
    distances_to_return = {}
    distances_to_return['distances_negative_median'] = torch.median(distances,dim=1).values.cpu()#torch.min(min_distances, dim=0).values.cpu()
    distances_to_return['distances_negative_min'] = min_distances.cpu()#torch.min(min_distances, dim=0).values.cpu()
    distances_to_return['distances_negative_sum'] = distances_negative_sum
    distances_to_return['distances_negative_sum_squared'] = distances_negative_sum_squared
    distances_to_return['total_count'] = total_count
    return distances_to_return

def find_distances_OOD_negatives(pos_cls_name, features_all_classes, args, gpu):
    pos_features = features_all_classes[pos_cls_name].to(f"cuda:{gpu}")
    neg_features = []
    chunk_size = 200
    min_distances = []
    distances_negative_sum = torch.tensor(0)
    distances_negative_sum_squared = torch.tensor(0)
    total_count = 0
    for cls in (list(args.other_negatives.keys())):
        neg_features.append(args.other_negatives[cls])
        if len(neg_features) == chunk_size:
            neg_features = torch.cat(neg_features).to(f"cuda:{gpu}")
            distances = pairwisedistances.__dict__[args.distance_metric](pos_features, neg_features)
            min_distances.append(torch.min(distances,dim=1).values.cpu())
            distances_negative_sum = distances_negative_sum + torch.sum(distances).cpu()
            distances_negative_sum_squared = distances_negative_sum_squared + torch.sum(distances**2).cpu()
            total_count += distances.nelement()
            neg_features = []
    if len(neg_features) > 0:
        neg_features = torch.cat(neg_features).to(f"cuda:{gpu}")
        distances = pairwisedistances.__dict__[args.distance_metric](pos_features, neg_features)
        min_distances.append(torch.min(distances,dim=1).values.cpu())
        distances_negative_sum = distances_negative_sum + torch.sum(distances).cpu()
        distances_negative_sum_squared = distances_negative_sum_squared + torch.sum(distances ** 2).cpu()
        total_count += distances.nelement()
    min_distances = torch.stack(min_distances,dim = 0)
    distances_to_return = {}
    distances_to_return['distances_negative_min'] = torch.min(min_distances, dim=1).values.cpu()
    distances_to_return['distances_negative_sum'] = distances_negative_sum
    distances_to_return['distances_negative_sum_squared'] = distances_negative_sum_squared
    distances_to_return['total_count'] = distances.nelement()
    return distances_to_return

def fit_high(distances, distance_multiplier, tailsize):
    if tailsize<1:
        tailsize = min(tailsize*distances.shape[1], distances.shape[1])
    tailsize = int(min(tailsize,distances.shape[1]))
    mr = weibull.weibull()
    mr.FitHigh(distances.double() * distance_multiplier, tailsize, isSorted=False)
    mr.tocpu()
    return mr

def OpenMax(pos_cls_name, features_all_classes, args, gpu, models=None):
    features = features_all_classes[pos_cls_name].clone().to(f"cuda:{gpu}")
    MAV = torch.mean(features,dim=0).to(f"cuda:{gpu}")
    distances = pairwisedistances.__dict__[args.distance_metric](features, MAV[None,:])
    weibull_model = fit_high(distances.T, args.distance_multiplier, args.tailsize)
    model = {}
    model['exemplars'] = MAV.cpu()
    model['weibulls'] = weibull_model
    return model

def OpenMax_Inference(pos_cls_name, features_all_classes, args, gpu, models):
    features = features_all_classes[pos_cls_name].to(f"cuda:{gpu}")
    probs=[]
    for class_name in sorted(models.keys()):
        MAV = models[class_name]['MAV'].to(f"cuda:{gpu}")
        distances = pairwisedistances.__dict__[args.distance_metric](features, MAV[None, :])
        probs.append(1 - models[class_name]['weibull_model'].wscore(distances.cpu()))
    probs = torch.cat(probs,dim=1)
    return probs

def MultiModalOpenMax(pos_cls_name, features_all_classes, args, gpu, models=None):
    features = features_all_classes[pos_cls_name]#.clone()
    # clustering
    Clustering_Algo = getattr(clustering, args.Clustering_Algo)
    # print(f"Clustering_Algo {Clustering_Algo}")
    features = features.type(torch.FloatTensor)
    centroids, assignments = Clustering_Algo(features, K=min(features.shape[0],100), verbose=False, distance_metric=args.distance_metric)
    features = features.cuda()
    centroids = centroids.type(features.dtype)
    # from IPython import embed;embed();
    MAVs=[]
    wbFits=[]
    smallScoreTensor=[]
    for MAV_no in set(assignments.cpu().tolist())-set([-1]):
        # print (f"MAV_no {MAV_no} {len(set(assignments.tolist())-set([-1]))}")
        MAV = centroids[MAV_no,:][None,:].cuda()
        f = features[assignments == MAV_no].cuda()
        # print(f"MAV {MAV.shape} {f.shape}")
        distances = pairwisedistances.__dict__[args.distance_metric](f, MAV)
        # print(f"distances {torch.max(distances)} {torch.min(distances)}")
        weibull_model = fit_high(distances.T, args.distance_multiplier, args.tailsize)
        # print(f"weibull_model done")
        MAVs.append(MAV)
        wbFits.append(weibull_model.wbFits)
        smallScoreTensor.append(weibull_model.smallScoreTensor)
    # from IPython import embed;embed();
    wbFits=torch.cat(wbFits)
    MAVs=torch.cat(MAVs)
    smallScoreTensor=torch.cat(smallScoreTensor)
    mr = weibull.weibull(dict(Scale=wbFits[:,1],
                              Shape=wbFits[:,0],
                              signTensor=weibull_model.sign,
                              translateAmountTensor=None,
                              smallScoreTensor=smallScoreTensor))
    mr.tocpu()
    model = {}
    model['MAVs'] = MAVs.cpu()
    model['weibulls'] = mr
    return model

def MultiModalOpenMax_Inference(pos_cls_name, features_all_classes, args, gpu, models=None):
    test_cls_feature = features_all_classes[pos_cls_name].to(f"cuda:{gpu}")
    probs=[]
    for cls_no, cls_name in enumerate(sorted(models.keys())):
        distances = pairwisedistances.__dict__[args.distance_metric](test_cls_feature,
                                                                     models[cls_name]['MAVs'].cuda().double())
        probs_current_class = 1-models[cls_name]['weibulls'].wscore(distances)
        probs.append(torch.max(probs_current_class, dim=1).values)
    probs = torch.stack(probs,dim=-1).cpu()
    print(f"probs {probs}")
    return probs


def fit_low(distances, distance_multiplier, tailsize, gpu):
    mr = weibull.weibull()
    mr.FitLow(distances.double() * distance_multiplier, min(tailsize,distances.shape[1]), isSorted=False, gpu=gpu)
    return mr

def set_cover(mr_model, positive_distances, cover_threshold):
    # compute probabilities
    probabilities = mr_model.wscore(positive_distances)

    # threshold by cover threshold
    e = torch.eye(probabilities.shape[0]).type(torch.BoolTensor)
    thresholded = probabilities >= cover_threshold
    thresholded[e] = True
    del probabilities

    # greedily add points that cover most of the others
    covered = torch.zeros(thresholded.shape[0]).type(torch.bool)
    extreme_vectors = []
    covered_vectors = []

    while not torch.all(covered).item():
        sorted_indices = torch.topk(torch.sum(thresholded[:, ~covered], dim=1),
                                    len(extreme_vectors)+1,
                                    sorted=False,
                                    ).indices
        for indx, sortedInd in enumerate(sorted_indices.tolist()):
            if sortedInd not in extreme_vectors:
                break
        else:
            print(thresholded.device,"ENTERING INFINITE LOOP ... EXITING")
            break
        covered_by_current_ev = torch.nonzero(thresholded[sortedInd, :], as_tuple=False)
        covered[covered_by_current_ev] = True
        extreme_vectors.append(sortedInd)
        covered_vectors.append(covered_by_current_ev.to("cpu"))
    del covered
    extreme_vectors_indexes = torch.tensor(extreme_vectors)
    params = mr_model.return_all_parameters()
    scale = torch.gather(params["Scale"].to("cpu"), 0, extreme_vectors_indexes)
    shape = torch.gather(params["Shape"].to("cpu"), 0, extreme_vectors_indexes)
    smallScore = torch.gather(
        params["smallScoreTensor"][:, 0].to("cpu"), 0, extreme_vectors_indexes
    )
    extreme_vectors_models = weibull.weibull(dict(Scale=scale,
                                                  Shape=shape,
                                                  signTensor=params["signTensor"],
                                                  translateAmountTensor=params["translateAmountTensor"],
                                                  smallScoreTensor=smallScore))
    del params
    return (extreme_vectors_models, extreme_vectors_indexes, covered_vectors)

def EVM(pos_cls_name, features_all_classes, args, gpu, models=None):
    # Find positive class features
    positive_cls_feature = features_all_classes[pos_cls_name].cuda()
    chunk_size = 200
    if args.tailsize<=1:
        tailsize = args.tailsize*positive_cls_feature.shape[0]
    else:
        tailsize = args.tailsize
    tailsize = int(tailsize)

    all_neg_features = []
    temp = []
    print(f"negative classes {set(features_all_classes.keys())-set([pos_cls_name])}")
    print(f"negative classes {len(set(features_all_classes.keys())-set([pos_cls_name]))}")
    for cls_name in set(features_all_classes.keys())-set([pos_cls_name]):
        temp.append(features_all_classes[cls_name])
        if len(temp) == chunk_size:
            all_neg_features.append(torch.cat(temp))
            temp = []
    if len(temp)>0:
        all_neg_features.append(torch.cat(temp))

    negative_distances=[]
    for batch_no, neg_features in enumerate(all_neg_features):
        distances = pairwisedistances.__dict__[args.distance_metric](positive_cls_feature, neg_features.cuda())
        # Store bottom k distances from each batch to the cpu
        sortedTensor = torch.topk(distances,
                                  min(tailsize,distances.shape[1]),
                                  dim = 1,
                                  largest = False,
                                  sorted = True).values
        del distances
        negative_distances.append(sortedTensor.cpu())

    positive_distances = pairwisedistances.__dict__[args.distance_metric](positive_cls_feature, positive_cls_feature)
    # check if distances to self is zero
    e = torch.eye(positive_distances.shape[0]).type(torch.BoolTensor)
    assert torch.allclose(positive_distances[e].type(torch.FloatTensor), \
                          torch.zeros(positive_distances.shape[0]),atol=1e-03) == True, \
        "Distances of samples to themselves is not zero"
    sortedTensor = torch.cat(negative_distances, dim=1).to(f"cuda:{gpu}")
    # Perform actual EVM training
    try:
        weibull_model = fit_low(sortedTensor, args.distance_multiplier, tailsize, gpu)
    except:
        from IPython import embed;embed();
    extreme_vectors_models, extreme_vectors_indexes, covered_vectors = set_cover(weibull_model,
                                                                                 positive_distances.cuda(),
                                                                                 args.cover_threshold)
    # from IPython import embed;embed();
    extreme_vectors = torch.gather(positive_cls_feature, 0, extreme_vectors_indexes[:,None].cuda().repeat(1,positive_cls_feature.shape[1]))
    extreme_vectors_models.tocpu()
    extreme_vectors = extreme_vectors.cpu()

    print(f"extreme_vectors {extreme_vectors.shape}")
    model = {}
    model['extreme_vectors'] = extreme_vectors
    model['weibulls'] = extreme_vectors_models
    return model

def EVM_Inference(pos_cls_name, features_all_classes, args, gpu, models=None):
    test_cls_feature = features_all_classes[pos_cls_name].cuda()
    probs=[]
    for cls_no, cls_name in enumerate(sorted(models.keys())):
        distances = pairwisedistances.__dict__[args.distance_metric](test_cls_feature,
                                                                     models[cls_name]['extreme_vectors'].cuda())
        probs_current_class = models[cls_name]['weibulls'].wscore(distances)
        probs.append(torch.max(probs_current_class, dim=1).values)
    probs = torch.stack(probs,dim=-1).cpu()
    return probs
