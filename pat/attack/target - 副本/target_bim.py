import math
import paddle
import paddle.nn as nn

class BIMTargetAttack(object):
    def __init__(self, model, img, label, eps, alpha, criterion=None):
        self.model = model
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.CosineSimilarity()
        self.img = img
        self.label = label
        self.epsilon = eps
        self.alpha = alpha
        self.num_iters = math.ceil(min(self.epsilon + 4, 1.25 * self.epsilon))

    def attack(self):
        origin_tensor_img = paddle.to_tensor(self.img)
        tensor_img = paddle.to_tensor(self.img)
        tensor_label = paddle.to_tensor(self.label)
        for step in range(self.num_iters):
            tensor_img.stop_gradient = False
            predict = self.model(tensor_img)
            loss = self.criterion(predict, tensor_label)
            for param in self.model.parameters():
                param.clear_grad()
            loss.backward(retain_graph=True)
            grad = paddle.to_tensor(tensor_img.grad)
            delta = self.alpha * paddle.sign(grad)
            tensor_img = tensor_img + delta
            clip_delta = paddle.clip(tensor_img - origin_tensor_img, -self.epsilon, self.epsilon)
            tensor_img = origin_tensor_img + clip_delta
            tensor_img = paddle.to_tensor(tensor_img.detach().numpy())

        return tensor_img