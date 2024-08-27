from s4dd import S4forDenovoDesign
from argparse import ArgumentParser
import os
if __name__ == "__main__":
    parser = ArgumentParser('(Multitask) Regression')
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    #parser.add_argument("--full-data", type=str, default=os.environ["SM_CHANNEL_DATA_FULL"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--output",type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    args = parser.parse_args().__dict__

    # Create an S4 model with (almost) the same parameters as in the paper.
    s4 = S4forDenovoDesign(
        n_max_epochs=3,  # This is for only demonstration purposes. Set this to a (much) higher value for actual training. Default: 400.
        batch_size=64,  # This is for only demonstration purposes. The value in the paper is 2048.
        device="cuda",  # replace this with "cpu" if you don't have a CUDA-enabled GPU.
    )
    # Pretrain the model on a small subset of ChEMBL
    s4.train(
        training_molecules_path=f"{args['train']}/chemblv31/train.zip",
        val_molecules_path=f"{args['test']}/chemblv31/valid.zip",
    )

    # save the pretrained model
    s4.save(f"{args['model_dir']}")

    # Fine-tune the model on a small subset of bioactive molecules
    s4.train(
        training_molecules_path=f"{args['train']}/pkm2/train.zip",
        val_molecules_path=f"{args['train']}/pkm2/valid.zip",
    )

    # save the fine-tuned model
    s4.save(f"{args['model_dir']}")


    # Design new molecules
    designs, lls = s4.design_molecules(n_designs=128, batch_size=64, temperature=1)

    # Save the designs
    with open(f"{args.output}/designs.smiles", "w") as f:
        f.write("\n".join(designs))

    # Save the log-likelihoods of the designs
    with open(f"{args.output}/lls.txt", "w") as f:
        f.write("\n".join([str(ll) for ll in lls]))
