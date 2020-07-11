import torch
from collections import defaultdict

class Solver:
    def __init__(self, model, optimizer, criterion, writer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer

    def train(self, args, loader):
        metrics = defaultdict(float)
        for step, ((x_i, x_j), y, _) in enumerate(loader):
            if not args.supervised:
                x_i = x_i.cuda(non_blocking=True)
                x_j = x_j.cuda(non_blocking=True)

                # positive pair, with encoding
                loss = self.model(x_i, x_j)
                # h_i, h_j, z_i, z_j = self.model(x_i, x_j)
                # loss = self.criterion(z_i, z_j)

                if step > 0 and step % 20 == 0:
                    print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}")
            else:
                x_i = x_i.to(
                    self.device
                )  # x_i and x_j are identital in supervised case (dataloader)
                y = y.to(self.device)

                h_i, _, _, _ = self.model(x_i, x_i)
                loss = self.criterion(h_i, y)

                auc, ap = get_metrics(
                    args.domain, y.detach().cpu().numpy(), h_i.detach().cpu().numpy()
                )

                metrics["Tag_auc/train"] += auc
                metrics["Tag_ap/train"] += ap

                if step > 0 and step % 20 == 0:
                    print(
                        f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t AUC: {auc}\t AP: {ap}"
                    )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar("Loss/train_step", loss.item(), args.global_step)
            if args.supervised:
                self.writer.add_scalar("AUC/train_step", auc, args.global_step)
                self.writer.add_scalar("AP/train_step", ap, args.global_step)

            metrics["Loss/train"] += loss.item()
            args.global_step += 1

        for k, v in metrics.items():
            metrics[k] /= len(loader)
        return metrics

    def validate(self, args, loader):
        self.model.eval()
        metrics = defaultdict(float)
        with torch.no_grad():
            for step, ((x_i, x_j), y, _) in enumerate(loader):
                if not args.supervised:
                    x_i = x_i.to(args.device)
                    x_j = x_j.to(args.device)

                    # positive pair, with encoding
                    # _, _, z_i, z_j = self.model(x_i, x_j)
                    loss = self.model(x_i, x_j)
                    # loss = self.criterion(z_i, z_j)
                else:
                    x_i = x_i.to(
                        self.device
                    )  # x_i and x_j are identital in supervised case (dataloader)
                    y = y.to(self.device)
                    h_i, _, _, _ = self.model(x_i, x_i)
                    loss = self.criterion(h_i, y)

                if step > 0 and step % 10 == 0:
                    print(
                        f"Step [{step}/{len(loader)}]\t Validation/Test Loss: {loss.item()}"
                    )

                metrics["Loss/test"] += loss.item()

        self.model.train()
        for k, v in metrics.items():
            metrics[k] /= len(loader)
        return metrics
