import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from pytorch_lightning.callbacks import Callback


class PlotSpectogramCallback(Callback):
    def on_train_start(self, trainer, pl_module):

        if not pl_module.hparams.time_domain:
            x, y = trainer.train_dataloader.dataset[0]

            fig = plt.figure()
            x_i = x[0, :]
            fig.add_subplot(1, 2, 1)
            plt.imshow(x_i)
            if x.shape[0] > 1:
                x_j = x[1, :]
                fig.add_subplot(1, 2, 2)
                plt.imshow(x_j)

            trainer.logger.experiment.add_figure(
                "Train/spectogram_sample", fig, global_step=0
            )
            plt.close()
