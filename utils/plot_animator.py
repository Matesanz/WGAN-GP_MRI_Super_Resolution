from matplotlib.animation import ImageMagickWriter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class PlotAnimator(ImageMagickWriter):
        
        def __init__(self, fig, fps=24, name='animation'):

                super(PlotAnimator, self).__init__(fps=fps)

                self.fig = fig

                self.setup(self.fig, name+'.gif', dpi=100)

                sns.set(color_codes=True, style='ticks', palette='colorblind')

                sns.despine(self.fig)
                plt.show(block=False)

                self.g_losses, self.d_losses = [], []


        def update_distribution_plot(self, ax, real_data, generated_data, epoch):

                ax.clear()
                ax = sns.regplot(x=real_data[:, 0], y=real_data[:, 1], color='orangered', ax=ax, order=2,
                                  label='Real Data Distribution', scatter_kws={'s': 7})
                ax = sns.regplot(x=generated_data[:, 0], y=generated_data[:, 1], color='dodgerblue', order=2, ax=ax,
                                  label='Fake Data Distribution', scatter_kws={'s': 7, 'alpha': 0.5})
                ax.set_title('Distributions')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_ylim([-0.1, 1.1])
                ax.set_xlim([-1.1, 1.1])
                ax.legend(loc='upper center', frameon=True)


        def update_training_plot(self, ax, generator_loss, discriminator_loss, epoch):

                self.g_losses.append(generator_loss)
                self.d_losses.append(discriminator_loss)

                epochs = np.linspace(0, epoch+1, len(self.g_losses))

                ax.clear()
                ax = sns.lineplot(epochs, self.g_losses, ax=ax, label='Generator Loss', color='g', linewidth=2.0)
                ax = sns.lineplot(epochs, self.d_losses, ax=ax, label='Discriminator Loss', color='orange', linewidth=2.0)
                ax.legend(loc='upper right', frameon=True)
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.set_ylim([-0.1, 1.1])
                ax.set_title('GAN loss')

        def epoch_end(self):

                self.fig.tight_layout(rect=[0, 0, 1, 0.90])
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                self.grab_frame()

        def close(self, last_frame_sec):

                [self.grab_frame() for _ in range(last_frame_sec * self.fps)]
                self.finish()
