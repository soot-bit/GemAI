from autogluon.tabular import TabularPredictor, TabularDataset

from ..config import settings, get_project_root
from ..data import load_split_data
from .. import utils


def run_autogluon_training():
    """
    Trains an AutoGluon model based on the project configuration.
    """
    utils.logger.info("Starting AutoGluon training...")

    train_df, val_df = load_split_data()
    train_data = TabularDataset(train_df)
    val_data = TabularDataset(val_df)
    utils.logger.info("Data prepared.")

    model_dir = get_project_root() / settings.paths.autogluon_dir

    predictor = TabularPredictor(
        label="price_bwp",
        eval_metric="root_mean_squared_error",
        path=model_dir,
    ).fit(
        train_data,
        presets="best_quality",
        auto_stack=False,
        time_limit=settings.autogluon.time_limit,
    )

    utils.logger.info(
        "--- AutoGluon Leaderboard (Model Ranking on Validation Data) ---"
    )
    leaderboard = predictor.leaderboard(val_data, silent=True)
    utils.logger.info(f"\n{leaderboard.to_string()}")

    # save leaderboard
    leaderboard_path = model_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path)
    utils.logger.info(f"Leaderboard saved to {leaderboard_path}")

    return predictor
