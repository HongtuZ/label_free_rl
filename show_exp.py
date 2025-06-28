import os
from pathlib import Path

import click
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

import rl.pytorch_utils as ptu
from rl.algorithm.certain import CERTAINArgs
from rl.net.agent import CertainAgent, ClassifierAgent, FocalAgent, UnicornAgent

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
ptu.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_certain_loss(
    title,
    xs,
    ys,
    pred_losses,
    recon_errors,
    save_path=None,
    cbar_ranges=None,
    task_goals=[],
):
    plt.figure(figsize=(10, 5))
    plt.suptitle(title, fontsize=14, y=0.99)
    ret_cbar_ranges = []

    for i, (data, label) in enumerate(
        zip([pred_losses, recon_errors], ["Predict Loss", "Recon Error"]), 1
    ):
        plt.subplot(1, 2, i)
        plt.xlim(-1.2, 1.2)  # 调整 x 轴范围
        plt.ylim(-1.2, 1.2)
        plt.title(label)
        plt.gca().add_artist(plt.Circle((0, 0), 1, color="gray", fill=False))
        plt.plot(0, 0, marker="*", markersize=6, color="green", markerfacecolor="none")
        for task_goal in task_goals:
            plt.plot(
                *task_goal.reshape(-1),
                marker="*",
                markersize=6,
                color="purple",
                markerfacecolor="none",
            )
        plt.scatter(xs, ys, c=data, cmap="coolwarm", s=1)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.gca().axis("off")
        cbar = plt.colorbar(orientation="horizontal", pad=0.00, fraction=0.1)
        if cbar_ranges:
            cbar.mappable.set_clim(*cbar_ranges[i - 1])
        cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, num=2))
        cbar.ax.set_xticklabels([f"{tick:.3f}" for tick in cbar.get_ticks()])
        ret_cbar_ranges.append((cbar.vmin, cbar.vmax))

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Figure saved to {save_path}")
    return ret_cbar_ranges


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config_path", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--checkpoint_dir", required=True, type=click.Path(exists=True, file_okay=False)
)
@click.option("--step", default=None, type=int, help="Step to load the checkpoint from")
def show_uncertainty(config_path, checkpoint_dir, step):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    certain_args = CERTAINArgs(**config["CERTAINArgs"])
    dataset = ptu.OfflineMetaDataset(str(certain_args.dataset_dir))

    step = step or max(
        int(p.stem.split("_")[-1]) for p in Path(checkpoint_dir).glob("*.pth")
    )
    context_agent_path = Path(checkpoint_dir) / f"context_agent_{step}.pth"
    certain_agent_path = Path(checkpoint_dir) / f"certain_agent_{step}.pth"

    certain_agent = CertainAgent.load_from(certain_agent_path)
    exp_name = Path(checkpoint_dir).parent.name.split("_")[0].split("-")
    if len(exp_name) != 2:
        raise ValueError(
            "Should be xxx-certain. Please check the checkpoint directory."
        )
    certain_args.context_agent_type = exp_name[0]

    context_agent_cls = {
        "focal": FocalAgent,
        "classifier": ClassifierAgent,
        "unicorn": UnicornAgent,
    }.get(certain_args.context_agent_type)
    if not context_agent_cls:
        raise NotImplementedError(
            f"Context agent type {certain_args.context_agent_type} not implemented."
        )
    context_agent = context_agent_cls.load_from(context_agent_path)

    certain_agent.to(ptu.device)
    context_agent.to(ptu.device)

    # Calculate the certain loss mean and std
    all_data = dataset.all_data
    all_context = torch.cat(
        [
            all_data["observations"],
            all_data["actions"],
            all_data["rewards"],
            all_data["next_observations"],
        ],
        dim=-1,
    ).to(ptu.device)
    all_latent_zs = context_agent.infer_latent(all_context)
    all_pred_losses = certain_agent.predict_loss(all_context, all_latent_zs)
    all_recon_errors = certain_agent.recon_error(all_context, all_latent_zs)
    pred_loss_min = torch.min(all_pred_losses).item()
    pred_loss_mean = torch.mean(all_pred_losses).item()
    recon_error_min = torch.min(all_recon_errors).item()
    recon_error_mean = torch.mean(all_recon_errors).item()
    cbar_ranges = [
        (pred_loss_min, pred_loss_mean + pred_loss_mean - pred_loss_min),
        (recon_error_min, recon_error_mean + recon_error_mean - recon_error_min),
    ]

    # sample and plot
    data = dataset.sample_all(50)

    def compute_and_plot(
        title,
        observations,
        actions,
        rewards,
        next_observations,
        task_goals,
        save_name,
        cbar_ranges=None,
    ):
        context = torch.cat(
            [observations, actions, rewards, next_observations], dim=-1
        ).to(ptu.device)
        latent_zs = context_agent.infer_latent(context)
        pred_losses = ptu.get_numpy(
            certain_agent.predict_loss(context, latent_zs)
        ).reshape(-1)
        pred_losses[pred_losses < cbar_ranges[0][0]] = cbar_ranges[0][1]
        recon_errors = ptu.get_numpy(
            certain_agent.recon_error(context, latent_zs)
        ).reshape(-1)
        xs, ys = ptu.get_numpy(observations[..., 0]).reshape(-1), ptu.get_numpy(
            observations[..., 1]
        ).reshape(-1)
        return plot_certain_loss(
            title,
            xs,
            ys,
            pred_losses,
            recon_errors,
            save_path=str(Path(checkpoint_dir) / save_name),
            cbar_ranges=cbar_ranges,
            task_goals=task_goals,
        )

    task_goals = np.expand_dims(
        np.array([(np.cos(task), np.sin(task)) for task in dataset.tasks]), axis=1
    )
    compute_and_plot(
        "ID-task ID-context",
        data["observations"].to(ptu.device),
        data["actions"].to(ptu.device),
        data["rewards"].to(ptu.device),
        data["next_observations"].to(ptu.device),
        task_goals,
        f"id-task-id-context_{step}.png",
        cbar_ranges=cbar_ranges,
    )

    ood_observations, ood_actions, ood_next_observations = (
        -data["observations"].to(ptu.device),
        -data["actions"].to(ptu.device),
        -data["next_observations"].to(ptu.device),
    )
    ood_rewards = ptu.from_numpy(
        np.linalg.norm(ptu.get_numpy(ood_next_observations) - task_goals, axis=-1)
    ).unsqueeze(-1)
    compute_and_plot(
        "ID-task OOD-context",
        ood_observations,
        ood_actions,
        ood_rewards,
        ood_next_observations,
        task_goals,
        f"id-task-ood-context_{step}.png",
        cbar_ranges=cbar_ranges,
    )

    ood_task_goals = -task_goals
    rewards = ptu.from_numpy(
        np.linalg.norm(
            ptu.get_numpy(data["next_observations"]) - ood_task_goals, axis=-1
        )
    ).unsqueeze(-1)
    compute_and_plot(
        "OOD-task ID-sa",
        data["observations"].to(ptu.device),
        data["actions"].to(ptu.device),
        rewards,
        data["next_observations"].to(ptu.device),
        ood_task_goals,
        f"ood-task-id-sa_{step}.png",
        cbar_ranges=cbar_ranges,
    )


if __name__ == "__main__":
    # cli()
    show_uncertainty()
