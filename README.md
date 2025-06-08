# Swift-SRGAN
## Architecture
<p align="center"> <img src="https://github.com/Koushik0901/Swift-SRGAN/blob/master/image-samples/SwiftSRGAN-architecture.png" width="850" height="450"  /> </p>

## Training
1. install requirements with:
    `pip install -r requirements.txt`
3. Train the model by executing:
    ``` bash
    cd swift-srgan
    python train.py --upscale_factor 4 --crop_size 96 --num_epochs 100
    ```
    
4. To convert the generator model to torchscript, run 
``` bash
python optimize-graph.py --ckpt_path ./checkpoints/netG_4x_epoch100.pth.tar --save_path ./checkpoints/optimized_model.pt --device cuda
```
