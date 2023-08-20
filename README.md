# FastMRI SSM TEAM
SSM Team FastMRI result

## Branch
- Varnet_Attention
- recons_unet
- pushing_reuslt
  
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
<pre><code>python train.py --cascade 6 --chans 10 --sens_chans 7 -e 10 -n Attention_6_10_7_first</code></pre>
2.	lr 5e-4로 10 epoch train
<pre><code>python train.py --cascade 6 --chans 10 --sens_chans 7 -e 10 --ckpt-dir ../result/Attention_6_10_7_first -n Attention_6_10_7_second -l 5e-4</code></pre>
3.	lr 3e-4, 3 epoch마다 0.8배 감소 / batch 4 training
<pre><code>python train.py --cascade 6 --chans 10 --sens_chans 7 -e 10 --ckpt-dir ../result/Attention_6_10_7_second -n Attention_6_10_7_final -l 3e-4 --last-train 1</code></pre>
batch 조건과 0.8배 감소 조건은 last-train argument가 1이 될때 작동하도록 코드를 작성하였으며, train_part.py의 train_epoch에서 확인 가능합니다.

### Upscaling Attention Unet training
학습을 진행하기에 앞서 Varnet의 test file과 validation file로부터 reconstruction을 수행하여야 합니다. (Varnet_Attention branch에서 수행하여야 합니다.)\
정상적으로 작동하기 위해서는 repo 폴더 바깥의 result 폴더에 AttVarnet_cascade8, Attention_6_10_7_final 폴더가 있어야 합니다. 이 폴더들은 pushing_result branch에 있으므로, 이 폴더를 복사하여 사용하시면 됩니다.
<pre><code>cp -r ./result/AttVarnet_cascade8 ../result</code></pre>
<pre><code>cp -r ./result/Attention_6_10_7_final ../result</code></pre>

다음 4개의 코드를 실행하여 2개의 Varnet의 reconsturction 결과를 얻을 수 있습니다. 이 코드를 실행하면 repo 폴더 바깥에 reconsturct폴더가 생성됩니다.\
<pre><code>python testfile_reconstruct.py -n "AttVarnet_cascade8 --cascade 8 -o '../reconstruct_cascade8'</code></pre>
<pre><code>python testfile_reconstruct.py -n "AttVarnet_cascade8 --cascade 8 -o '../reconstruct_cascade8' --type val</code></pre>
<pre><code>python testfile_reconstruct.py -n "AttVarnet_6_10_7_final --cascade 6 --chans 10 --sens_chans 7 -o '../reconstruct_6_10_7'</code></pre>
<pre><code>python testfile_reconstruct.py -n "AttVarnet_6_10_7_final --cascade 6 --chans 10 --sens_chans 7 -o '../reconstruct_6_10_7' --type val</code></pre>

reconsturction이 완료되면 recons_unet branch 에서 학습을 진행합니다.\

38 epoch 학습을 진행하였습니다.\
<pre><code>python train.py --ckpt-dir '../result/Unet_32_1_high/' --last-train 1 --in-chans 3 -t '../reconstruct_6_10_7/train/image' -v '../reconstruct_6_10_7/val/image' --input-key recons --grappa-key grappa --target-key target -e 38 -r 200 -n Unet_32_1_high -t2 '../reconstruct_cascade8/train/image/' -v2 '../reconstruct_cascade8/val/image/'</code></pre>
이 코드를 실행하면 result에 Unet_32_1_high가 생성됩니다.

##	Evaluation

###	Varnet reconstruction 
Varnet_Attention branch에서 leaderboard reconstruction을 수행합니다. (8,9,4), (6,10,7) 조합에 대해서 각각 수행해야 합니다.

<pre><code>python reconstruct.py -n AttVarnet_cascade8</code></pre>
<pre><code>python reconstruct.py -n Attention_6_10_7_final</code></pre>
###	Upscaling Attention Unet restruction & forwarding
3	Evaluation
3.1	

## Reference
[1] Zbontar, J.*, Knoll, F.*, Sriram, A.*, Murrell, T., Huang, Z., Muckley, M. J., ... & Lui, Y. W. (2018). fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. arXiv preprint arXiv:1811.08839.

[2] Sriram, A.*, Zbontar, J.*, Murrell, T., Defazio, A., Zitnick, C. L., Yakubova, N., ... & Johnson, P. (2020). End-to-End Variational Networks for Accelerated MRI Reconstruction. In MICCAI, pages 64-73.

[3] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
