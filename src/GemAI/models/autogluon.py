from autogluon.tabular import TabularPredictor, TabularDataset

from autogluon.tabular import TabularPredictor, TabularDataset

from ..config import config, get_project_root
from ..data import load_data
from .. import utils


def train_ag():
    """
    Trains an AutoGluon model based on the project configuration.
    """
    utils.log_header("AUTOGLUON TRAINING")
    utils.log("Starting AutoGluon training...")

    trn_df, val_df = load_data()
    trn_data = TabularDataset(trn_df)
    val_data = TabularDataset(val_df)
    utils.log("Data prepared.")

    mdl_dir = get_project_root() / config.dir.logs / config.dir.autogluon
    mdl_dir.mkdir(parents=True, exist_ok=True)  # Ensure model directory exists

    predictor = TabularPredictor(
        label="price_bwp",
        eval_metric="root_mean_squared_error",
        path=mdl_dir,
    ).fit(
        trn_data,
        num_gpus=0,
        fit_strategy="sequential",
        auto_stack=False,
        num_bag_folds=0,  # prevents _BAG models
        num_stack_levels=0,  # no _L2 or _L3 models
        dynamic_stacking=False,  # disable the stacking logic
        time_limit=config.autogluon.timeout,
    )

    utils.log_header("AUTOGLUON LEADERBOARD")
    leaderboard = predictor.leaderboard(val_data, silent=False)
    utils.log(f"\n{leaderboard.to_string()}")

    # save leaderboard
    lb_path = mdl_dir / "leaderboard.csv"
    leaderboard.to_csv(lb_path)
    utils.log(f"Leaderboard saved to {lb_path}")
    summary = predictor.fit_summary()
    print(summary["model_performance"])
    utils.log_header("AUTOGLUON TRAINING COMPLETED")

    return predictor
