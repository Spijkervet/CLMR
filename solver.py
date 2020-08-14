import torch
import logging
from collections import defaultdict
from utils.eval import get_metrics
import time

class Solver:
    def __init__(self, model, optimizer, criterion, writer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer

    def train(self, args, loader):
        metrics = defaultdict(float)
        avg_time = 0
        for step, ((x_i, x_j), y) in enumerate(loader):
            t0 = time.time()
            if not args.supervised:
                x_i = x_i.to(args.device)
                x_j = x_j.to(args.device)

                # positive pair, with encoding
                h_i, h_j, z_i, z_j = self.model(x_i, x_j)
                loss = self.criterion(z_i, z_j)

                if self.writer and step > 0 and step % 100 == 0:
                    logging.info(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()} Time: {avg_time / step}")
                    self.writer.add_scalar("Loss/train_step", loss, args.global_step)
            else:
                x_i = x_i.to(
                    args.device
                )  # x_i and x_j are identital in supervised case (dataloader)
                y = y.to(args.device)
        
                if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    h_i, _ = self.model.module.get_latent_representations(x_i)
                else:
                    h_i, _ = self.model.get_latent_representations(x_i)

                loss = self.criterion(h_i, y)

                auc, ap = get_metrics(
                    args.domain, y.detach().cpu().numpy(), h_i.detach().cpu().numpy()
                )

                metrics["AUC_tag/train"] += auc
                metrics["AP_tag/train"] += ap

                if self.writer and step > 0 and step % 20 == 0:
                    logging.info(
                        f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t AUC: {auc}\t AP: {ap}\t Time: {avg_time / step}"
                    )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            metrics["Loss/train"] += loss.item()

            avg_time += (time.time() - t0)
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
                    if self.writer and step > 0 and step % 20 == 0:
                        logging.info(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}")
                else:
                    x_i = x_i.to(
                        args.device
                    )  # x_i and x_j are identital in supervised case (dataloader)
                    y = y.to(args.device)

                    if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                        h_i, _ = self.model.module.get_latent_representations(x_i)
                    else:
                        h_i, _ = self.model.get_latent_representations(x_i)

                    loss = self.criterion(h_i, y)

                    auc, ap = get_metrics(
                        args.domain, y.detach().cpu().numpy(), h_i.detach().cpu().numpy()
                    )
                    metrics["AUC_tag/test"] += auc
                    metrics["AP_tag/test"] += ap
                    if self.writer and step > 0 and step % 20 == 0:
                        logging.info(
                            f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t AUC: {auc}\t AP: {ap}"
                        )
                        
                metrics["Loss/test"] += loss.item()

        self.model.train()
        for k, v in metrics.items():
            metrics[k] /= len(loader)
        return metrics
