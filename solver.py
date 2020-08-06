import torch
import logging
from collections import defaultdict

class Solver:
    def __init__(self, model, optimizer, criterion, writer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer

    def train(self, args, loader):
        metrics = defaultdict(float)
        for step, ((x_i, x_j), y) in enumerate(loader):
            if not args.supervised:
                x_i = x_i.cuda(non_blocking=True)
                x_j = x_j.cuda(non_blocking=True)

                # positive pair, with encoding
                h_i, h_j, z_i, z_j = self.model(x_i, x_j)
                loss = self.criterion(z_i, z_j)

                if self.writer and step > 0 and step % 1 == 0:
                    logging.info(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}")
            else:
                x_i = x_i.to(
                    args.device
                )  # x_i and x_j are identital in supervised case (dataloader)
                y = y.to(args.device)

                loss = self.model(x_i, x_i)

                auc, ap = get_metrics(
                    args.domain, y.detach().cpu().numpy(), h_i.detach().cpu().numpy()
                )

                metrics["AUC_tag/train"] += auc
                metrics["AP_tag/train"] += ap

                if self.writer and step > 0 and step % 20 == 0:
                    logging.info(
                        f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t AUC: {auc}\t AP: {ap}"
                    )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.writer:
                self.writer.add_scalar("Loss/train_step", loss.item(), args.global_step)
                if args.supervised:
                    self.writer.add_scalar("AUC_tag/train_step", auc, args.global_step)
                    self.writer.add_scalar("AP_tag/train_step", ap, args.global_step)

            metrics["Loss/train"] += loss.item()
            args.global_step += 1

        for k, v in metrics.items():
            metrics[k] /= len(loader)
        return metrics

    def validate(self, args, loader):
        self.model.eval()
        metrics = defaultdict(float)
        with torch.no_grad():
            for step, ((x_i, x_j), y) in enumerate(loader):
                if not args.supervised:
                    x_i = x_i.to(args.device)
                    x_j = x_j.to(args.device)

                    # positive pair, with encoding
                    # loss = self.model(x_i, x_j)
                    h_i, h_j, z_i, z_j  = self.model(x_i, x_j)
                    loss = self.criterion(z_i, z_j)
                else:
                    x_i = x_i.to(
                        args.device
                    )  # x_i and x_j are identital in supervised case (dataloader)
                    y = y.to(args.device)
                    loss = self.model(x_i, x_i)

                if self.writer and step > 0 and step % 10 == 0:
                    logging.info(
                        f"Step [{step}/{len(loader)}]\t Validation/Test Loss: {loss.item()}"
                    )

                metrics["Loss/test"] += loss.item()

        self.model.train()
        for k, v in metrics.items():
            metrics[k] /= len(loader)
        return metrics
