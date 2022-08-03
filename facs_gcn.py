from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from facs_model import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset_str', default='cora', type=str)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--reward_class_2',type=float, default=1)
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--epoch',type=int,default=10)
    parser.add_argument('--alpha',type=float,default=.5)
    args = parser.parse_args()

    r11 = 1
    r12 = 0
    r21 = 0
    r22 = args.reward_class_2
    epsilon = 0
    delta = 1. / (r11 + r22 - r12 - r21)
    alpha = args.alpha
    a = r11 - r12
    b = r22 - r21
    R = np.array([r11, r12, r21, r22])
    R1 = torch.FloatTensor([[r11, r12], [r12, r11]])
    R2 = torch.FloatTensor([[r22, r21], [r21, r22]])
    classes = np.arange(2)
    seed = args.seed
    print('---------------------------------------------')
    print(f'-----------Seed: {seed}----------------------')
    path = 'data/'
    adj, features, labels, gender, train_mask, val_mask, test_mask = load_data(path=path, dataset=args.dataset_str, seed=seed)

    args.n_feat = features.shape[1]
    args.n_classes = 2
    best_auc = -1
    reg_lambda=5e-3
    nclasses =2
    model = AdaGCN(features.shape[1], args.hidden_dim, 2, dropout=args.dropout, dropout_adj=.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    adv = GCN(2, 2, 2, dropout=args.dropout, dropout_adj=.2)
    optimizer_adv = torch.optim.Adam(adv.parameters(), lr=.01, weight_decay=args.weight_decay)

    sample_weights = torch.ones(torch.sum(train_mask))
    sample_weights = sample_weights / sample_weights.sum()
    results = torch.zeros(adj.shape[0], 2)
    for layer in range(args.L):

        for epoch in range(args.epoch):
            optimizer.zero_grad()

            output = model(features,adj,train_mask)
            loss = F.nll_loss(output[train_mask],labels[train_mask],reduction='none')
            loss = loss*sample_weights
            preds = torch.argmax(output,dim=1)
            loss = loss.sum()
            l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))


            output1 = F.softmax(output, dim=1)
            pred_inv = torch.ones(output.shape)
            pred_inv[:, 1] = output1[:, 0]
            pred_inv[:, 0] = output1[:, 1]

            r_ratio = torch.ones(output.shape)
            for i in range(r_ratio.shape[0]):
                r_ratio[i, 0] = a / b
                r_ratio[i, 1] = b / a

            preds = output / pred_inv
            preds = r_ratio * preds
            h = delta * torch.log(preds)

            adv_model = adv(h.detach(),adj)
            adv_loss_temp = F.nll_loss(adv_model[train_mask],gender[train_mask],reduction='sum')

            loss = ((1-alpha)*loss-(alpha*adv_loss_temp.detach())) + reg_lambda / 2 * l2_reg
            loss.backward()
            optimizer.step()

            adv.requires_grad_(True)
            optimizer_adv.zero_grad()
            adv_model = adv(h.detach(), adj)
            adv_loss = F.nll_loss(adv_model[train_mask], gender[train_mask], reduction='sum')
            adv_loss.backward()
            optimizer_adv.step()

            #eval
            if epoch % 200 ==0:
                model.eval()
                pred_ = output.max(1)[1].type_as(labels)
                acc_train = acc_measurements(output[train_mask], labels[train_mask], gender[train_mask])
                auc_roc_train, auc_m, auc_f = auc_measurements(output[train_mask], labels[train_mask],
                                                               gender[train_mask])

                parity, equality = fair_metric(pred_[train_mask].numpy(), labels[train_mask].numpy(),
                                               gender[train_mask].numpy())
                print(
                    f'|*|Epoch Hyperparamter{epoch}||Training: acc layer #{layer}: {acc_train[0]} || Auc layer #{layer}: {auc_roc_train} || AUC Male: {auc_m} || AUC Female: {auc_f} || SP: {parity} || EQ: {equality}|*|')

                output = model(features, adj, val_mask)
                pred_ = output.max(1)[1].type_as(labels)
                acc_train = acc_measurements(output[val_mask], labels[val_mask], gender[val_mask])
                auc_roc_train, auc_m, auc_f = auc_measurements(output[val_mask], labels[val_mask],
                                                               gender[val_mask])

                parity, equality = fair_metric(pred_[val_mask].numpy(), labels[val_mask].numpy(),
                                               gender[val_mask].numpy())
                print(
                    f'|*|Epoch Hyperparamter{epoch}||Validation: acc layer #{layer}: {acc_train[0]} || Auc layer #{layer}: {auc_roc_train} || AUC Male: {auc_m} || AUC Female: {auc_f} || SP: {parity} || EQ: {equality}|*|')

                output = model(features, adj, test_mask)
                pred_ = output.max(1)[1].type_as(labels)
                acc_train = acc_measurements(output[test_mask], labels[test_mask], gender[test_mask])
                auc_roc_train, auc_m, auc_f = auc_measurements(output[test_mask], labels[test_mask],
                                                               gender[test_mask])

                parity, equality = fair_metric(pred_[test_mask].numpy(), labels[test_mask].numpy(),
                                               gender[test_mask].numpy())
                print(
                    f'|*|Epoch Hyperparamter{epoch}||Test: acc layer #{layer}: {acc_train[0]} || Auc layer #{layer}: {auc_roc_train} || AUC Male: {auc_m} || AUC Female: {auc_f} || SP: {parity} || EQ: {equality}|*|')

        model.eval()
        output = model(features, adj, torch.arange(adj.shape[0])).detach()
        output = F.softmax(output,dim=1)
        pred_inv = torch.ones(output.shape)
        pred_inv[:, 1] = output[:, 0]
        pred_inv[:, 0] = output[:, 1]

        r_ratio = torch.ones(output.shape)
        for i in range(r_ratio.shape[0]):
            r_ratio[i, 0] = a / b
            r_ratio[i, 1] = b / a

        preds = output / pred_inv
        preds = r_ratio * preds
        h = alpha * torch.log(preds)
        output_logp = torch.log(output)
        results += h

        y_codes = np.array([-1. , 1.])
        y = labels.detach().numpy()[train_mask]
        y_coding = y_codes.take(classes == y[:, np.newaxis])
        # y_i^T * R (reward matrices)
        coding = []
        h =h[train_mask]
        for i in range(len(y_coding)):
            if y_coding[i][0] == 1:
                coding.append(np.dot(y_coding[i], R1).dot(h[i].T))
            else:
                coding.append(np.dot(y_coding[i], R2).dot(h[i].T))

        y_coding = torch.FloatTensor(np.array(coding))
        estimator_weight = (-1. / 2) * y_coding

        # update sample weight
        sample_weights*= torch.exp(estimator_weight)
        sample_weights = sample_weights / sample_weights.sum()
        sample_weights = sample_weights.detach()

        # update features
        features = SparseMM.apply(adj, features).detach()

    pred_ = results.max(1)[1].type_as(labels)
    acc_train = acc_measurements(results[train_mask], labels[train_mask], gender[train_mask])
    auc_roc_train, auc_m, auc_f = auc_measurements(results[train_mask], labels[train_mask],
                                                   gender[train_mask])

    parity, equality = fair_metric(pred_[train_mask].numpy(), labels[train_mask].numpy(),
                                   gender[train_mask].numpy())
    print(
        f'|*|Final--------Epoch Hyperparamter{epoch}||Training: acc: {acc_train[0]} || Auc: {auc_roc_train} || AUC Male: {auc_m} || AUC Female: {auc_f} || SP: {parity} || EQ: {equality}|*|')

    pred_ = results.max(1)[1].type_as(labels)
    acc_train = acc_measurements(results[val_mask], labels[val_mask], gender[val_mask])
    auc_roc_train, auc_m, auc_f = auc_measurements(results[val_mask], labels[val_mask],
                                                   gender[val_mask])

    parity, equality = fair_metric(pred_[val_mask].numpy(), labels[val_mask].numpy(),
                                   gender[val_mask].numpy())
    print(
        f'|*|Final--------Epoch Hyperparamter{epoch}||Validation: acc: {acc_train[0]} || Auc: {auc_roc_train} || AUC Male: {auc_m} || AUC Female: {auc_f} || SP: {parity} || EQ: {equality}|*|')


    pred_ = results.max(1)[1].type_as(labels)
    acc_train = acc_measurements(results[test_mask], labels[test_mask], gender[test_mask])
    auc_roc_train, auc_m, auc_f = auc_measurements(results[test_mask], labels[test_mask],
                                                   gender[test_mask])

    parity, equality = fair_metric(pred_[test_mask].numpy(), labels[test_mask].numpy(),
                                   gender[test_mask].numpy())

    print(
        f'|*|Final --------Epoch Hyperparamter{epoch}||Test: acc: {acc_train[0]} || Auc: {auc_roc_train} || AUC Male: {auc_m} || AUC Female: {auc_f} || SP: {parity} || EQ: {equality}|*|')
    print(confusion_matrix(labels[test_mask].numpy(), pred_[test_mask].numpy()))
    print(precision_recall_fscore_support(labels[test_mask].numpy(), pred_[test_mask].numpy(), average='binary'))



