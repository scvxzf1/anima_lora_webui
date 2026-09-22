"""Bounded real Krea OOM recovery with fixed numerical precision."""

from bench.adaptive_runtime.recovery import main as recovery_main


def main():
    recovery_main(model_family="krea2")


if __name__ == "__main__":
    main()
