# ContentVec with no Fairseq

This repository provides standalone inference codes of the [ContentVec](https://github.com/auspicious3000/contentvec) model, which is based on Fairseq. 

Since Fairseq is hard to install on python>=3.10, this project aims to isolate contentvec from it. This standalone version **only requires PyTorch** as its dependency.

This model is tested on `pytorch2.5`, `python3.12`

## Preparation:

To use ContentVec without Fairseq, the original checkpoint needs to be converted. Follow these steps:

### Step1: Download the original ContentVec checkpoint

Download the original contentvec checkpoint, eg: [contentvec768l12](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt) or [contentvec256l9](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr). These can be found in repositories like [Sovits](https://github.com/svc-develop-team/so-vits-svc/) or [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC/).

### Step2: Run the Conversion Script

Then run the script below to convert the checkpoint, replace the `/path/to/original_chackpoint.pt` to the path of downloaded checkpoint, and replace the `/path/to/target_checkpoint.pt` to where you want to save the converted checkpoint.

```bash
python convert_checkpoint.py --original_checkpoint_path /path/to/original_chackpoint.pt --target_checkpoint_path /path/to/target_checkpoint.pt
```

**Why Convert the Checkpoint?**

The original checkpoint relies on Fairseq for loading. Without Fairseq, the checkpoint cannot be unpickled using `torch.load()`! This conversion script removes unnecessary metadata, retaining only the configuration and model state dictionary. The converted checkpoint can then be safely loaded with `weights_only=True`

## Usage

### Test the Converted Model

Please refer to `test_api.py` for an example of how to use the converted model to extract the feature from an audio.

### (Optional) Verify Conversion Results

If Fairseq is installed, use `test_fairseq.py` to compare the outputs of the converted checkpoint and the original checkpoint to ensure consistency.

## Example

### Sovits 4.1

Copy the `./contentvec` folder to [the vencoder folder](https://github.com/svc-develop-team/so-vits-svc/tree/4.1-Stable/vencoder).

Rename the `example_sovits.py` to `ContentVec768L12.py` and replace [the original one](https://github.com/svc-develop-team/so-vits-svc/blob/4.1-Stable/vencoder/ContentVec768L12.py) with the renamed file..

## Disclaimer

Any organization or individual is prohibited from using any technology in this repo to generate or edit someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.