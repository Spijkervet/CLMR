import torch
import numpy as np
from utils import tagwise_auc_ap


class Supervised:
    def __init__(self, args, model):

        self.model = model
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.supervised_lr,
            weight_decay=1e-6,
            momentum=0.9,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=2, verbose=True
        )

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def solve(self, train_loader, test_loader, start_epoch, epochs):
        for epoch in range(start_epoch, epochs):
            for step, (x, y, _) in enumerate(train_loader):
                self.optimizer.zero_grad()
                x = x.to(args.device)
                y = y.to(args.device)

                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

                auc, ap = tagwise_auc_ap(
                    y.cpu().detach().numpy(), output.cpu().detach().numpy()
                )

                print(
                    self.metrics(step, train_loader, loss.item(), auc.mean(), ap.mean())
                )

            eval_loss, eval_auc, eval_ap = self.eval(test_loader, args.device)
            print(
                "Eval:",
                self.metrics(
                    step, test_loader, eval_loss.item(), eval_auc.mean(), eval_ap.mean()
                ),
            )

            self.scheduler.step(eval_loss)
            curr_lr = self.optimizer.param_groups[0]["lr"]
            print("Learning rate : {}".format(curr_lr))
            if curr_lr < 1e-7:
                print("Early stopping")
                break

    def eval(self, test_loader, device):
        losses = []
        ys = []
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for step, (x, y, _) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)

                output = self.model(x)
                loss = self.criterion(output, y)
                losses.append(loss.mean().detach().cpu().numpy())

                ys.extend(y.detach().cpu().numpy())
                outputs.extend(output.detach().cpu().numpy())

        auc, ap = tagwise_auc_ap(np.array(ys), np.array(outputs))
        losses = np.array(losses).mean()
        self.model.train()
        return losses, auc, ap

    def metrics(self, step, train_loader, loss, auc, ap):
        return "[{}/{}] Loss: {}, AUC: {}, AP: {}".format(
            step, len(train_loader), loss, auc, ap
        )
