
from datetime import datetime
from typing import Callable, List
import numpy as np
import os
import logging
import time
import torch
from environment.env import SimulationEnvironment
from loss.loss import Loss
from model.pinn import PINN
from train.params import SimulationParameters
from visualization.plotting import plot_all, plot_running_average
from visualization.report import create_report

logger = logging.getLogger()


class Training:
    def __init__(self, model: PINN,
                 loss: Loss,
                 environment: SimulationEnvironment,
                 initial_condition: Callable) -> None:
        self.model = model
        self.loss = loss
        self.environment = environment
        self.initial_condition = initial_condition
        self.params = SimulationParameters()

        self.best_loss = float("inf")

    def start(self):
        start = time.time()
        self.create_run_directory()

        logging.info(f"Starting training run: {self.params.RUN_NUM}")
        loss_total, loss_r, loss_i, loss_b = self.train()
        losses = (loss_total, loss_r, loss_i, loss_b)
        logging.info("Finished training")

        logging.info("Visualizing results")
        self.print_summary(loss_total[-1], loss_r[-1], loss_i[-1], loss_b[-1])

        if self.params.VISUALIZE:
            os.makedirs(os.path.join(self.params.DIR,
                        f"run_{self.params.RUN_NUM}", "img"), exist_ok=True)
            self.plot_averages(losses)
            self.visualize_results()

        if self.params.REPORT:
            self.report(loss_total[-1], loss_r[-1], loss_i[-1], loss_b[-1], time.time()-start)

        return self.model, loss_total, loss_r, loss_i, loss_b

    def train(self):
        lbfgs_optimizer = torch.optim.LBFGS(self.model.parameters(
        ), lr=1.0, history_size=100, line_search_fn="strong_wolfe", max_iter=20)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.LEARNING_RATE)
        loss_values = []
        residual_loss_values = []
        initial_loss_values = []
        boundary_loss_values = []

        # Define the closure function for L-BFGS
        def closure():
            loss = self.loss(self.model, epoch)
            optimizer.zero_grad()
            loss[0].backward()
            return loss[0]

        for epoch in range(self.params.EPOCHS):
            try:

                if self.params.OPTIM_SWITCH and epoch >= self.params.OPTIM_SWITCH:
                    total_loss = lbfgs_optimizer.step(closure).item()
                    loss = self.loss(self.model, epoch)  # TODO optimize this
                    optimizer.zero_grad()
                else:
                    loss = self.loss(self.model, epoch)
                    optimizer.zero_grad()
                    loss[0].backward()
                    total_loss = loss[0].item()

                    if self.params.CLIP_GRAD:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                    optimizer.step()

                if self.params.SAVE_BEST_CLB:
                    self.save_best_callback(total_loss)

                loss_values.append(total_loss)
                residual_loss_values.append(loss[1].item())
                initial_loss_values.append(loss[2].item())
                boundary_loss_values.append(loss[3].item())

                if (epoch + 1) % 1000 == 0:
                    self.print_epoch_report(epoch, loss)

            except KeyboardInterrupt:
                break

        return np.array(loss_values), \
            np.array(residual_loss_values), \
            np.array(initial_loss_values), \
            np.array(boundary_loss_values)

    def visualize_results(self):
        save_path = os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}")
        plot_all(save_path,
                 self.model,
                 self.environment,
                 self.initial_condition,
                 limit=0.5,
                 limit_wave=self.params.PEAK_HEIGHT)

    def plot_averages(self, losses):
        save_path = os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}")
        plot_running_average(save_path, losses[0], "Loss function (running average)", "total_loss")
        plot_running_average(
            save_path, losses[1], "Residual loss function (running average)", "residual_loss")
        plot_running_average(
            save_path, losses[2], "Initial loss function (running average)", "initial_loss")
        plot_running_average(
            save_path, losses[3], "Boundary loss function (running average)", "boundary_loss")

    def print_summary(self, total_loss, initial_loss, residual_loss, boundary_loss):
        logger.info(f'Total loss: \t{total_loss:.5f} ({total_loss:.3E})')
        logger.info(f'Interior loss: \t{initial_loss:.5f} ({initial_loss:.3E})')
        logger.info(f'Initial loss: \t{residual_loss:.5f} ({residual_loss:.3E})')
        logger.info(f'Boundary loss: \t{boundary_loss:.5f} ({boundary_loss:.3E})')

    def print_epoch_report(self, epoch: int, loss: List[float]):
        logger.info(f"Epoch: {epoch + 1} - \
            Loss: {float(loss[0].item()):>7f}, \
            Residual Loss: {float(loss[1].item()):>7f}, \
            Initial Loss: {float(loss[2].item()):>7f}, \
            Boundary Loss: {float(loss[3].item()):>7f}")

    def create_run_directory(self):
        try:
            os.makedirs(os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}"), exist_ok=True)
            logger.info("Run directory created successfully")
        except OSError as error:
            logger.error(f"Run directory creation failed: {error}")

    def save_best_callback(self, loss: float):
        if loss < self.best_loss:
            torch.save(self.model, os.path.join(self.params.DIR,
                       f"run_{self.params.RUN_NUM}", f"best_{self.params.RUN_NUM}.pt"))

            self.best_loss = loss

    def report(self, loss_total, loss_r, loss_i, loss_b, time=0.0):
        logger.info("Creating report")

        date = datetime.now()
        save_dir = self.params.DIR
        num = self.params.RUN_NUM
        context = {
            'num': num,
            'date': date.strftime("%Y-%m-%d %H:%M:%S"),
            'weight_r': self.params.INITIAL_WEIGHT_RESIDUAL,
            'weight_i': self.params.INITIAL_WEIGHT_INITIAL,
            'weight_b': self.params.INITIAL_WEIGHT_BOUNDARY,
            'layers': self.params.LAYERS,
            'neurons': self.params.NEURONS_PER_LAYER,
            'epochs': self.params.EPOCHS,
            'lr': self.params.LEARNING_RATE,
            'total_loss': f"{loss_total:.3E}",
            "residual_loss": f"{loss_r:.3E}",
            "initial_loss": f"{loss_i:.3E}",
            "boundary_loss": f"{loss_b:.3E}",
            "img_loss": os.path.join(save_dir, f"run_{num}", "total_loss.png"),
            "img_loss_r": os.path.join(save_dir, f"run_{num}", "residual_loss.png"),
            "img_loss_i": os.path.join(save_dir, f"run_{num}", "initial_loss.png"),
            "img_loss_b": os.path.join(save_dir, f"run_{num}", "boundary_loss.png"),
            "mesh_name": self.params.MESH if self.params.MESH else "-",
            "loss_name": self.params.LOSS,
            "optim_name": self.params.OPTIM_SWITCH,
            "base_height": self.params.BASE_HEIGHT,
            "decay_rate": self.params.DECAY_RATE,
            "peak_height": self.params.PEAK_HEIGHT,
            "x_divisor": self.params.X_DIVISOR,
            "y_divisor": self.params.Y_DIVISOR,
            "time": time
        }

        report_path = os.path.join(
            self.params.DIR, f"run_{self.params.RUN_NUM}", f"report_{self.params.RUN_NUM}.pdf")
        create_report(context,
                      env_path='.',
                      template_path='report_template.html',
                      report_title=report_path)
