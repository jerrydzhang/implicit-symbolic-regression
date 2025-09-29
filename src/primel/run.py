import argparse
import json
import numpy as np
from experiments.base_experiment import Experiment


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment.")
    parser.add_argument(
        "--name", type=str, required=True, help="Name of the experiment."
    )
    parser.add_argument(
        "--description", type=str, default="", help="Description of the experiment."
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the training data (.npy file).",
    )

    # All other arguments will be collected into a dictionary for the 'parameters'
    # Example: --learning_rate 0.01 --batch_size 32
    # This was implemented by parsing the arguments and then creating a dictionary
    # from the remaining arguments.
    args, remaining_argv = parser.parse_known_args()

    # Parse the remaining arguments as key-value pairs
    parameters = {}
    for i in range(0, len(remaining_argv), 2):
        key = remaining_argv[i].lstrip("-")
        value = remaining_argv[i + 1]
        try:
            # Convert to number if possible
            if "." in value:
                parameters[key] = float(value)
            else:
                parameters[key] = int(value)
        except ValueError:
            parameters[key] = value

    # Load training data
    try:
        train_data = np.load(args.train_data_path)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {args.train_data_path}")
        return

    # Create and run the experiment
    experiment = Experiment(
        name=args.name,
        description=args.description,
        train_data=train_data,
        parameters=parameters,
    )

    print(f"Starting experiment {args.name} with parameters: {parameters}")
    experiment.prepare()
    experiment.run()
    experiment.save()
    print(f"Finished experiment {args.name}.")


if __name__ == "__main__":
    main()
