
from datetime import datetime
from typing import List
import numpy as np
import os
import time
import torch
from conditions.initial import initial_condition
from environment.domain import Domain
from environment.env import SimulationEnvironment
from loss.loss import Loss
from model.pinn import PINN
from model.weights import Weights
from train.params import SimulationParameters
from visualization.plotting import create_gif, plot_initial_condition, plot_running_average, plot_simulation_by_frame
from visualization.report import create_report


class Training:
    def __init__(self, model: PINN, loss: Loss, params: SimulationParameters, environment: SimulationEnvironment, weights: Weights) -> None:
        self.model = model
        self.loss = loss

        self.weights = weights
        self.params = params
        self.environment = environment

        self.best_loss = float("inf")
    

    def start(self):
        start = time.time()
        run_num = self.params.RUN_NUM
        save_path = self.params.DIR
        self.create_run_directory()
        print("Starting run: ", run_num)

        loss_total, loss_r, loss_i, loss_b = self.train()

        self.print_summary()
        plot_running_average(save_path, loss_total, "Loss function (running average)", "total_loss", run_num)
        plot_running_average(save_path, loss_r, "Residual loss function (running average)", "residual_loss", run_num)
        plot_running_average(save_path, loss_i, "Initial loss function (running average)", "initial_loss", run_num)
        plot_running_average(save_path, loss_b, "Boundary loss function (running average)", "boundary_loss", run_num)

        if self.params.VISUALIZE:
            self.visualize_results()

        if self.params.REPORT:
            losses = self.loss.verbose(self.model)
            self.report(losses, time.time()-start)

        return self.model, loss_total, loss_r, loss_i, loss_b


    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.LEARNING_RATE)
        loss_values = []
        residual_loss_values = []
        initial_loss_values = []
        boundary_loss_values = []

        for epoch in range(self.params.EPOCHS):
            try:
                loss: torch.Tensor = self.loss(self.model)
                optimizer.zero_grad()
                loss[0].backward()
                if self.params.CLIP_GRAD:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()

                if self.params.SAVE_BEST_CLB:
                    self.save_best_callback(loss[0].item())

                loss_values.append(loss[0].item())
                residual_loss_values.append(loss[1].item())
                initial_loss_values.append(loss[2].item())
                boundary_loss_values.append(loss[3].item())

                if (epoch + 1) % 1000 == 0:
                    self.print_epoch_report(epoch, loss)

            except KeyboardInterrupt:
                break

        return np.array(loss_values), np.array(residual_loss_values), np.array(initial_loss_values), np.array(boundary_loss_values)
    
    def evaluate(self):
        self.create_run_directory()
        self.train()
        self.print_summary()
        self.visualize_results()


    def visualize_results(self):
        mesh = self.params.MESH
        run_num = self.params.RUN_NUM
        save_path = self.params.DIR
        plot_initial_condition(save_path, self.environment, self.model, initial_condition, run_num, mesh=mesh)
        plot_simulation_by_frame(save_path, self.model, self.environment, run_num, mesh=mesh)
        create_gif(save_path, run_num, self.environment.domain.T_DOMAIN[1]) 
        

    def print_summary(self):
        losses = self.loss.verbose(self.model)

        print(f'Total loss: \t{losses[0]:.5f} ({losses[0]:.3E})')
        print(f'Interior loss: \t{losses[1]:.5f} ({losses[1]:.3E})')
        print(f'Initial loss: \t{losses[2]:.5f} ({losses[2]:.3E})')
        print(f'Boundary loss: \t{losses[3]:.5f} ({losses[3]:.3E})')


    def print_epoch_report(self, epoch: int, loss: List[float]):
        print(f"Epoch: {epoch + 1} - \
            Loss: {float(loss[0].item()):>7f}, \
            Residual Loss: {float(loss[1].item()):>7f}, \
            Initial Loss: {float(loss[2].item()):>7f}, \
            Boundary Loss: {float(loss[3].item()):>7f}")
    

    def create_run_directory(self):
        try:
            os.makedirs(os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}", "img"), exist_ok=True)
            print(f"Run directory created successfully.")
        except OSError as error:
            print(f"Run directory creation failed: {error}")
    

    def save_best_callback(self, loss: float):
        if loss < self.best_loss:
            torch.save(self.model, os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}", f"best_{self.params.RUN_NUM}.pt"))
            self.best_loss = loss

    def report(self, losses, time=0.0):
        date = datetime.now()

        context = {
                'num': self.params.RUN_NUM,
                'date': date.strftime("%Y-%m-%d %H:%M:%S"),
                'weight_r': self.weights.WEIGHT_RESIDUAL, 
                'weight_i': self.weights.WEIGHT_INITIAL, 
                'weight_b': self.weights.WEIGHT_BOUNDARY,
                'layers': self.params.LAYERS, 
                'neurons': self.params.NEURONS_PER_LAYER,
                'epochs': self.params.EPOCHS, 
                'lr': self.params.LEARNING_RATE,
                'total_loss': f"{losses[0]:.3E}",
                "residual_loss": f"{losses[1]:.3E}",
                "initial_loss": f"{losses[2]:.3E}",
                "boundary_loss": f"{losses[3]:.3E}",
                "img_loss": os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}", "total_loss.png"),
                "img_loss_r": os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}", "residual_loss.png"),
                "img_loss_i": os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}", "initial_loss.png"),
                "img_loss_b": os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}", "boundary_loss.png"),
                "mesh_name": self.params.MESH if self.params.MESH else "-",
                "time": time
            }

        report_path = os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}", f"report_{self.params.RUN_NUM}.pdf")
        create_report(context,
                    env_path='.',
                    template_path='report_template.html',
                    report_title=report_path)
