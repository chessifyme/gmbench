from bench.config import ReviewEnv
from bench.paths import DATA_PUBLIC_SHEETS, DATA_PUBLIC_TASKS, PREVIOUS_WEIGHTS_FILE
from review_app.data_dirs import seed_data_dir


def main() -> None:
    data_dir = ReviewEnv().review_data_dir
    if data_dir is None:
        raise SystemExit("REVIEW_DATA_DIR is not set: seeding is only for a deployed review app with its own data folder")
    try:
        copied = seed_data_dir(data_dir=data_dir, source_tasks=DATA_PUBLIC_TASKS, source_sheets=DATA_PUBLIC_SHEETS, source_previous_weights=PREVIOUS_WEIGHTS_FILE)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"seeded {len(copied)} new file(s) into {data_dir}")


if __name__ == "__main__":
    main()
