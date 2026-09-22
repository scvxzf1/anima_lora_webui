"""Bounded real Z-Image OOM recovery with fixed numerical precision."""

from bench.adaptive_runtime.recovery import main as recovery_main


def main():
    recovery_main(model_family="z_image")


if __name__ == "__main__":
    main()
