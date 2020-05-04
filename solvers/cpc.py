import torch
from model import save_model
from modules import NT_Xent
from validation.audio.latent_representations import audio_latent_representations

class CPC:
    def __init__(self, args, model, optimizer, scheduler, writer):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.device = args.device

    def solve(self, args, train_loader, test_loader, start_epoch, epochs):
        validate_idx = 5
        for epoch in range(start_epoch, epochs):
            lr = self.optimizer.param_groups[0]["lr"]
            if epoch % validate_idx == 0:
                self.visualise_latent_space(args, train_loader, test_loader)

            loss_epoch = self.train(args, train_loader)
            self.writer.add_scalar("Loss/train", loss_epoch, epoch)
            self.writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch}\t lr: {round(lr, 5)}")

            # validate
            if epoch % validate_idx == 0:
                print("Validation")
                test_loss_epoch = self.test(args, test_loader)
                self.writer.add_scalar("Loss/test", test_loss_epoch, epoch)

            if self.scheduler:
                self.scheduler.step()

            if epoch % 10 == 0:
                save_model(args, self.model, self.optimizer, name="cpc")

            args.current_epoch += 1

    def train(self, args, train_loader):
        loss_epoch = 0
        for step, ((x, _), y, _) in enumerate(train_loader):
            self.optimizer.zero_grad()
            x = x.to(self.device)

            loss, output, _, _ = self.model(x, y)
            # loss = loss.mean()  # accumulate loss for all GPUs

            loss.backward()
            self.optimizer.step()

            if step % 1 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

            self.writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            loss_epoch += loss.item()
            args.global_step += 1

        return loss_epoch / len(train_loader)

    def test(self, args, loader):
        self.model.eval()
        loss_epoch = 0
        with torch.no_grad():
            for step, ((x, _), y, _) in enumerate(loader):
                x = x.to(args.device)

                loss, output, _, _ = self.model(x, y)
                loss = loss.mean()  # accumulate loss for all GPUs 

                if step % 1 == 0:
                    print(f"Step [{step}/{len(loader)}]\t Test Loss: {loss.item()}")

                loss_epoch += loss.item()
        self.model.train()
        return loss_epoch / len(loader)

    def visualise_latent_space(self, args, train_loader, test_loader):
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