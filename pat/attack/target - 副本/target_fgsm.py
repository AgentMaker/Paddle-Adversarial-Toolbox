import paddle
import paddle.nn as nn

class FGSMTargetAttack(object):
    def __init__(self, model, img, label, eps, criterion=None):
        self.model = model
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.CosineSimilarity()
        self.img = img
        self.label = label
        self.epsilon = eps

    def attack(self):
        tensor_img = paddle.to_tensor(self.img)
        tensor_label = paddle.to_tensor(self.label)
        tensor_img.stop_gradient = False
        predict = self.model(tensor_img)
        loss = self.criterion(predict, tensor_label)
        for param in self.model.parameters():
            param.clear_grad()
        loss.backward(retain_graph=True)
        grad = paddle.to_tensor(tensor_img.grad)
        grad = paddle.sign(grad)
        tensor_img = tensor_img + self.epsilon * grad
        tensor_img = paddle.to_tensor(tensor_img.detach().numpy())

        return tensor_img