"""
Entry point for simple experiments.

Currently just delegates to train_and_predict.main()
so you can run:

  python -m app.main
"""

from app.train_and_predict import main as train_and_predict_main


def main():
    train_and_predict_main()


if __name__ == "__main__":
    main()
