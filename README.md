# FastMRI SSM TEAM
SSM Team FastMRI result

## Branch
- Varnet_Attention
- Unet_Attention

## Training
### Attention Varnet training (Varnet_Attention branch)
#### (8,9,4) Varnet training
3 epoch을 먼저 학습한 다음, best checkpoint로부터 10 epoch을 더 학습하면 best_model_Var_8_9_4.pt를 얻을 수 있습니다.\
첫번째 3 epoch 학습:
<pre><code>python train.py --cascade 8 -n Attention_8 -e 3</code></pre>
두번째 10 epoch 학습:
<pre><code>python train.py –cascade 8 -n Attention_8 --ckpt-dir ../result/Attention_8 -e 10</code></pre>
--ckpt-dir을 통해 학습을 시작할 checkpoint를 설정할 수 있습니다.

### (6,10,7) Varnet training
3단계의 걸쳐서 학습을 수행합니다.
1.	lr 1e-3로 10 epoch 학습:
<pre><code>python train.py --cascade 6 --chans 10 --sens-chans 7 -e 10 -n Attention_6_10_7_first</code></pre>
2.	lr 5e-4로 10 epoch train
<pre><code>python train.py --cascade 6 --chans 10 --sens-chans 7 -e 10 --ckpt-dir ../result/Attention_6_10_7_first -n Attention_6_10_7_second -l 5e-4</code></pre>
3.	lr 3e-4, 3 epoch마다 0.8배 감소 / batch 4 training
<pre><code>python train.py --cascade 6 --chans 10 --sens-chans 7 -e 10 --ckpt-dir ../result/Attention_6_10_7_second -n Attention_6_10_7_last -l 3e-4 --last-train 1</code></pre>
batch 조건과 0.8배 감소 조건은 last-train argument가 1이 될때 작동하도록 코드를 작성하였으며, train_part.py의 train_epoch에서 확인 가능합니다.

1.2	Upscaling Attention Unet training
2	Forwarding
2.1	Varnet forwarding
2.1.1	(8,9,4) Varnet forwarding
2.1.2	(6,10,7) Varnet forwarding
2.2	Upscaling Attention Unet forwarding
3	Evaluation
3.1	

## Reference
[1] Zbontar, J.*, Knoll, F.*, Sriram, A.*, Murrell, T., Huang, Z., Muckley, M. J., ... & Lui, Y. W. (2018). fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. arXiv preprint arXiv:1811.08839.

[2] Sriram, A.*, Zbontar, J.*, Murrell, T., Defazio, A., Zitnick, C. L., Yakubova, N., ... & Johnson, P. (2020). End-to-End Variational Networks for Accelerated MRI Reconstruction. In MICCAI, pages 64-73.

[3] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
