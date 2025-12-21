from autogluon.tabular import TabularPredictor, TabularDataset

from ..config import config, get_project_root
from ..data import load_split_data
from .. import utils


def run_autogluon_training():
    """
    Trains an AutoGluon model based on the project configuration.
    """
    utils.log_rule("AUTOGLUON TRAINING")
    utils.log_info("Starting AutoGluon training...")

    train_df, val_df = load_split_data()
    train_data = TabularDataset(train_df)
    val_data = TabularDataset(val_df)
    utils.log_info("Data prepared.")

    model_dir = get_project_root() / config.paths.log_dir / config.autogluon.dir
    model_dir.mkdir(parents=True, exist_ok=True)  # Ensure model directory exists

    # classical_models = {
    #     "GBM": {},  # LightGBM
    #     "CAT": {},  # CatBoost
    #     "XGB": {},  # XGBoost
    #     "RF": {},  # Random Forest
    #     "XT": {},  # Extra Trees
    #     "LR": {},  # Linear Regression (Ridge/Logistic)
    # }
    #
    predictor = TabularPredictor(
        label="price_bwp",
        eval_metric="root_mean_squared_error",
        path=model_dir,
    ).fit(
        train_data,
        # hyperparameters=classical_models,
        num_gpus=0,
        # presets="best_quality",
        fit_strategy="sequential",
        auto_stack=False,
        num_bag_folds=0,  # prevents _BAG models
        num_stack_levels=0,  # no _L2 or _L3 models
        dynamic_stacking=False,  # disable the stacking logic
        time_limit=config.autogluon.time_limit,
    )

    utils.log_rule("AUTOGLUON LEADERBOARD")
    leaderboard = predictor.leaderboard(val_data, silent=False)
    utils.log_info(f"\n{leaderboard.to_string()}")

    # save leaderboard
    leaderboard_path = model_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path)
    utils.log_info(f"Leaderboard saved to {leaderboard_path}")
    summary = predictor.fit_summary()
    print(summary["model_performance"])
    utils.log_rule("AUTOGLUON TRAINING COMPLETED")

    return predictor
