from s4dd import S4forDenovoDesign
from argparse import ArgumentParser
import os
if __name__ == "__main__":
    parser = ArgumentParser('(Multitask) Regression')
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    #parser.add_argument("--full-data", type=str, default=os.environ["SM_CHANNEL_DATA_FULL"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    args = parser.parse_args().__dict__

    # Create an S4 model
    s4 = S4forDenovoDesign(
        n_max_epochs=3,  # This is for only demonstration purposes. Set this to a (much) higher value for actual training. Default: 400
        batch_size=64,  # This is for only demonstration purposes. The value in the paper is 2048.
        device="cuda",  # replace this with "cpu" if you don't have a CUDA-enabled GPU
    )
    # Pretrain the model on a small subset of ChEMBL
    s4.train(
        training_molecules_path=f"{args['train']}/train.zip",
        val_molecules_path=f"{args['test']}/valid.zip",
    )
    # Save the model
    s4.save(args['model_dir'])
