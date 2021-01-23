import torch
import torch.nn as nn
import numpy as np
from vast import losses
import torch.utils.data as data_util
from vast.tools import logger as vastlogger
torch.manual_seed(0)

logger = vastlogger.get_logger()

class MLP(nn.Module):
    def __init__(self, input_feature_size=2048, num_classes=50):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_feature_size, out_features=1024, bias=True)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class netowrk():
    def __init__(self, num_classes):
        self.net = MLP(num_classes=num_classes)
        self.net = self.net.cuda()
        self.cls_names = []

    def modify_net(self, new_num_classes):
        new_net = MLP(num_classes=new_num_classes).cuda()
        weights = self.net.state_dict()
        to_add = torch.rand(new_net.fc2.out_features - self.net.fc2.out_features, weights['fc2.weight'].shape[1]).cuda()
        weights['fc2.weight'] = torch.cat((weights['fc2.weight'], to_add))
        to_add = torch.rand(new_net.fc2.out_features - self.net.fc2.out_features).cuda()
        weights['fc2.bias'] = torch.cat((weights['fc2.bias'], to_add))
        new_net.load_state_dict(weights)
        self.net = new_net

    def prep_training_data(self, training_data):
        classes_in_consideration = self.cls_names + sorted(list(set(training_data.keys())-set(self.cls_names)))
        if len(classes_in_consideration)!=self.net.fc2.out_features:
            logger.critical(f"New number of classes {len(classes_in_consideration)}")
            self.modify_net(len(classes_in_consideration))
        training_tensor_x=[]
        training_tensor_label=[]
        for cls in training_data:
            training_tensor_x.append(training_data[cls])
            training_tensor_label.extend([cls]*training_data[cls].shape[0])
        training_tensor_x = torch.cat(training_tensor_x).type(torch.FloatTensor).cuda()
        training_tensor_label = np.array(training_tensor_label)
        training_tensor_y=torch.zeros(training_tensor_label.shape[0]).type(torch.LongTensor).cuda()
        for cls_no,cls in enumerate(classes_in_consideration):
            training_tensor_y[training_tensor_label==cls]=cls_no
        self.dataset = data_util.TensorDataset(training_tensor_x, training_tensor_y)
        self.cls_names = classes_in_consideration

    def training(self, training_data, epochs=150, lr=0.01, batch_size=256):
        self.prep_training_data(training_data)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        loader = data_util.DataLoader(self.dataset, batch_size=batch_size)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        no_of_print_statements=10
        printing_interval=epochs//no_of_print_statements
        for epoch in range(epochs):
            loss_history=[]
            # numer of correct, total number
            train_accuracy = torch.zeros(2, dtype=int)
            for x, y in loader:
                optimizer.zero_grad()
                output = self.net(x)
                loss = loss_fn(output, y)
                train_accuracy += losses.accuracy(output, y)
                loss_history.extend(loss.tolist())
                loss.mean().backward()
                optimizer.step()

            to_print=f"Epoch {epoch:03d}/{epochs:03d} \t"\
                     f"train-loss: {np.mean(loss_history):1.5f}  \t"\
                     f"accuracy: {float(train_accuracy[0]) / float(train_accuracy[1]):9.5f}"
            if epoch%printing_interval==0:
                logger.info(to_print)
            else:
                logger.debug(to_print)

    def inference(self, validation_data):
        results = {}
        for cls in validation_data:
            with torch.no_grad():
                logits = self.net(validation_data[cls].type(torch.FloatTensor).cuda()).cpu()
                results[cls] = torch.nn.functional.softmax(logits, dim=1)
        return results




