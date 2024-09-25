import torch

class SaliencyMap:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.model.zero_grad()
        self.device = device

    def calculate_saliency_map(self, image, target_class=None):
        image = image.to(self.device)
        image.requires_grad = True
        output = self.model(image)
        if target_class is None:
            target_class = output.max(1)[1].item()
        target = torch.tensor([target_class]).to(self.device)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        saliency_map = image.grad.abs().max(1)[0]
        return saliency_map
    