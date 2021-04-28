import math
import paddle

class PGDAttack(object):
    def __init__(self, model, criterion, img, label, eps, alpha, num_iters=6):
        self.model = model
        self.criterion = criterion
        self.img = img
        self.label = label
        self.epsilon = eps
        self.alpha = alpha
        self.num_iters = num_iters

    def attack(self):
        origin_tensor_img = paddle.to_tensor(self.img)
        tensor_img = paddle.to_tensor(self.img)
        tensor_label = paddle.to_tensor(self.label)
        delta_init = paddle.uniform(self.img.shape, dtype='float32', min=-self.epsilon, max=self.epsilon)
        tensor_img = tensor_img + delta_init
        clip_delta = paddle.clip(tensor_img - origin_tensor_img, -self.epsilon, self.epsilon)
        tensor_img = origin_tensor_img + clip_delta

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