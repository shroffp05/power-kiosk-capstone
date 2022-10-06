"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Khuyen Tran 
"""

# import hydra
# from omegaconf import DictConfig
# from hydra.utils import to_absolute_path as abspath
import argparse

"""
@hydra.main(config_path="../config", config_name="main")
def train_model(config: DictConfig):

    input_path = abspath(config.processed.path)
    output_path = abspath(config.final.path)

    print(f"Train modeling using {input_path}")
    print(f"Model used: {config.model.name}")
    print(f"Save the output to {output_path}")
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass a contract location ID")

    parser.add_argument("--cl", type=int, help="Type a contract location ID")
    parser.add_argument(
        "--cls",
        type=str,
        help="Type a comma separated string of contract location IDs",
    )
    parser.add_argument("--a", type=int, help="For all contract location IDs")

    args = parser.parse_args()

    print(args.cl)
    print(args.cls)
    print(args.a)

    arg_vals = [args.cl, args.cls, args.a]

    assert (
        sum(1 for _ in arg_vals if _ is not None) == 1
    ), "Make sure only one argument is passed at a time"
