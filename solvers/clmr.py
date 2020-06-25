import torch
import os
import numpy as np
from model import save_model
from modules import NT_Xent
from validation.audio.latent_representations import audio_latent_representations
from utils.eval import get_metrics, tagwise_auc_ap
from modules.pytorchtools import EarlyStopping
from utils.optimizer import set_learning_rate

class CLMR:
    def __init__(self, args, model, optimizer, scheduler, writer):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.device = args.device

        if args.supervised:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = NT_Xent(args.batch_size, args.temperature, args.device)

    def solve(self, args, train_loader, val_loader, test_loader, start_epoch, epochs):
        validate_idx = 1
        latent_idx = 5
        avg_test_idx = 5
        save_idx = 1

        self.optimizer = set_learning_rate(self.optimizer, args.learning_rate)
        
        for epoch in range(start_epoch, epochs):
            if epoch % latent_idx == 0:
                self.visualise_latent_space(args, train_loader, test_loader)

            learning_rate = self.optimizer.param_groups[0]['lr']
            print("Learning rate : {}".format(learning_rate))

            loss_epoch, auc_epoch, ap_epoch = self.train(args, train_loader)
            self.writer.add_scalar("Loss/train", loss_epoch, epoch)
            self.writer.add_scalar("AUC/train", auc_epoch, epoch)
            self.writer.add_scalar("AP/train", ap_epoch, epoch)
            self.writer.add_scalar("Misc/learning_rate", learning_rate, epoch)
            print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch}\t lr: {round(learning_rate, 5)}")


            # test
            if args.dataset == "magnatagatune":
                # validate
                if epoch % validate_idx == 0:
                    validate_loss_epoch = self.validate(args, val_loader)
                    self.writer.add_scalar("Loss/validation", validate_loss_epoch, epoch)

            if epoch % avg_test_idx == 0:
                print("Testing average scores")

                if args.supervised:
                    test_loss_epoch, test_auc_epoch, test_ap_epoch = self.test_avg(args, test_loader)
                    self.writer.add_scalar("AUC/test", test_auc_epoch, epoch)
                    self.writer.add_scalar("AP/test", test_ap_epoch, epoch)
                else:
                    test_loss_epoch = self.validate(args, test_loader)

                self.writer.add_scalar("Loss/test", test_loss_epoch, epoch)

            if args.supervised:
                if self.scheduler:
                    self.scheduler.step(validate_loss_epoch)
                    curr_lr = self.optimizer.param_groups[0]['lr']
                    print("Adjusted learning rate : {}".format(curr_lr))
                    if curr_lr < 1e-6:
                        print("Early stopping")

            if epoch % save_idx == 0:
                save_model(args, self.model, self.optimizer, name=args.model_name)

            args.current_epoch += 1

        

    def train(self, args, train_loader):
        loss_epoch = 0
        auc_epoch = 0
        ap_epoch = 0
        for step, ((x_i, x_j), y, _) in enumerate(train_loader):
            self.optimizer.zero_grad()

            if not args.supervised:
                x_i = x_i.to(self.device)
                x_j = x_j.to(self.device)

                # positive pair, with encoding
                h_i, z_i = self.model(x_i)
                h_j, z_j = self.model(x_j)
                loss = self.criterion(z_i, z_j)
                
                if step % 20 == 0:
                    print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")
            else:
                x_i = x_i.to(self.device) # x_i and x_j are identital in supervised case (dataloader)
                y = y.to(self.device)

                h_i, _ = self.model(x_i)
                loss = self.criterion(h_i, y)
                
                auc, ap = get_metrics(args.domain, y, h_i)
                auc_epoch += auc
                ap_epoch += ap
                
                if step % 20 == 0:
                    print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}\t AUC: {auc}\t AP: {ap}")

            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar("Loss/train_step", loss.item(), args.global_step)

            if args.supervised:
                self.writer.add_scalar("AUC/train_step", auc, args.global_step)
                self.writer.add_scalar("AP/train_step", ap, args.global_step)

            loss_epoch += loss.item()
            args.global_step += 1

        return loss_epoch / len(train_loader), auc_epoch / len(train_loader), ap_epoch / len(train_loader)

    def validate(self, args, loader):
        self.model.eval()
        loss_epoch = 0
        with torch.no_grad():
            for step, ((x_i, x_j), y, _) in enumerate(loader):
                if not args.supervised:
                    x_i = x_i.to(args.device)
                    x_j = x_j.to(args.device)

                    # positive pair, with encoding
                    h_i, z_i = self.model(x_i)
                    h_j, z_j = self.model(x_j)

                    loss = self.criterion(z_i, z_j)
                else:
                    x_i = x_i.to(self.device) # x_i and x_j are identital in supervised case (dataloader)
                    y = y.to(self.device)
                    h_i, _ = self.model(x_i)
                    loss = self.criterion(h_i, y)


                if step % 10 == 0:
                    print(f"Step [{step}/{len(loader)}]\t Validation/Test Loss: {loss.item()}")

                loss_epoch += loss.item()

        self.model.train()
        return loss_epoch / len(loader)

    def test_avg(self, args, loader):
        self.model.eval()
        loss_epoch = 0
        pred_array = []
        id_array = []
        tracks = loader.dataset.tracks_list_test
        with torch.no_grad():
            for step, (track_id, fp, y, _) in enumerate(tracks):
                if not args.supervised:
                    x_i = x_i.to(self.device)
                    x_j = x_j.to(self.device)

                    # positive pair, with encoding
                    h_i, z_i = self.model(x_i)
                    h_j, z_j = self.model(x_j)
                    loss = self.criterion(z_i, z_j)
                else:
                    x = loader.dataset.get_full_size_audio(track_id, fp)
                    x = x.to(self.device)
                    y = y.to(self.device)

                    output, _ = self.model(x)

                    for b in output:
                        pred_array.append(b.detach().cpu().numpy())
                        id_array.append(track_id)
                    
                if step % 100 == 0:
                    print(f"Step [{step}/{len(tracks)}]")


        self.model.train()

        y_pred = []
        y_true = []
        pred_array = np.array(pred_array)
        id_array = np.array(id_array)
        for track_id, _, label, _ in tracks:
            # average over track
            avg = np.mean(pred_array[np.where(id_array == track_id)], axis=0)
            y_pred.append(avg)
            y_true.append(label.numpy())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        auc, ap = tagwise_auc_ap(y_true, y_pred)
        auc = auc.mean()
        ap = ap.mean()
        return loss_epoch / len(loader), auc, ap

    def visualise_latent_space(self, args, train_loader, test_loader):
        if args.model_name == "clmr":
            if args.domain == "audio":
                audio_latent_representations(
                    args,
                    train_loader.dataset,
                    self.model,
                    args.current_epoch,
                    0,
                    args.global_step,
                    self.writer,
                    train=True,
                )
                audio_latent_representations(
                    args,
                    test_loader.dataset,
                    self.model,
                    args.current_epoch,
                    0,
                    args.global_step,
                    self.writer,
                    train=False,
                )
            elif args.domain == "scores":
                vision_latent_representations(
                    args,
                    train_loader.dataset,
                    self.model,
                    self.optimizer,
                    args.current_epoch,
                    0,
                    args.global_step,
                    self.writer,
                    train=True,
                )
                vision_latent_representations(
                    args,
                    test_loader.dataset,
                    self.model,
                    self.optimizer,
                    args.current_epoch,
                    0,
                    args.global_step,
                    self.writer,
                    train=False,
                )
            else:
                raise NotImplementedError
