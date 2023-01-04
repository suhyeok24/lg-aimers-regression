# 🚘 자율주행 센서의 안테나 성능 예측 AI 경진대회 🚘

Regression을 통한 예측 정확도를 극대화해보자!

<br/><br/>

## Team

**Name** : 안장최

**Members** : 고려대학교 산업경영공학부 안영지, 장수혁, 최대원

<br/><br/>

## Project Descriptions

**Link** : [자율주행 센서의 안테나 성능 예측 AI 경진대회 (DACON)](https://dacon.io/competitions/official/235927/overview/description)


**[목표]**  
공정 데이터를 활용하여 Radar 센서의 안테나 성능 예측을 위한 AI 모델 개발 


**[배경]**  
Radar는 자율주행 차에 있어 차량과의 거리, 상대 속도, 방향 등을 측정해주는 필수적인 센서 부품 <br/>
LG에서는 제품의 성능 평가 공정에서 양품과 불량을 선별 중. <br/>
AI 기술을 활용하여 공정 데이터와 제품 성능간 상관 분석을 통해 제품의 불량을 예측/분석하고, 
수율을 극대화하여 불량으로 인한 제품 폐기 비용을 감축시키는 것이 목표.


**[주최]**  
LG AI Research 

**[평가 산식]**  
Normalized RMSE (NRMSE)
> def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(1,15): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    return score

<br/><br/>

## 👣 Score(Public)
RRDB+ : 23.40812(38th)

<br/><br/>

## 🌐 Environment
Colab Pro+  
GPU: A100-SXM4-40GB * 1(Main) , Tesla T4*1(Sub)

<br/><br/>

## 🔥 Competition Strategies

**1. Patches(for data augmentation)**  
- Train patches : original 1640 images → 26240(1640*16) patches (X4 downsampling, non-overlapping)  
- Test patches: original 18 images → 882(18*49) patches (X4 downsampling, overlapping(to remove border artifacts)) 

<br/>

**2. Data Transform**  
Non-destructive transformations (not to add or lose the information)
- Flip  
- Transpose  
- RandomRotate  
- ShiftScaleRotate  

<br/>

**3. Training Methods**
- EarlyStopping  
  > To prevent overfitting  
  > If validation loss does not improve after given patience(=2), training is earlystopped  
- Fine-tuning with pre-trained model  
  > pretrained model : [RRDB_PSNR_x4.pth](https://github.com/xinntao/ESRGAN/tree/master/models)(the PSNR-oriented model with high PSNR performance)  
  > Retraining entire model : Judging that the similarity between DF2K dataset(pretrained model) and our training datset is small  

<br/>

**4. Loss Function**
- L1 loss + L2 loss (2:1)
  > L2 loss : PSNR is based on MSE  
  > L1 loss: For better convergence [https://arxiv.org/pdf/1707.02921v1.pdf](https://arxiv.org/pdf/1707.02921v1.pdf)

<br/>

**5. Learning Scheduler, Optimizer**
- StepLR  
  > step_size = 3, gamma = 0.5  
  > Decays the learning rate of each parameter in half once per 3 epochs 
- Adam  

<br/>

**6. Post Processing**
- Geometric Self-Ensemble [https://arxiv.org/pdf/1707.02921v1.pdf](https://arxiv.org/pdf/1707.02921v1.pdf)
  > <img width="751" alt="KakaoTalk_Photo_2023-01-04-11-58-49" src="https://user-images.githubusercontent.com/55012723/210476203-015eac00-d0e0-4d10-8eb5-a772a9910097.png">


<br/><br/> 

## Main configuration & Hyperparameters
'''
1. Manuel_seed : 42  

2. Model :  
   > num_feat : 64 , Channel number of intermediate features.  
   > growth_channel: 32 , Channels for each growth(dense connection).  
   > num_block: 23 , number of RRDB blocks.  

3. Dataloader :  
   > train_batch_size : 4  
   > test_batch_size: 1  
   > num_workers: 4   

4. Train :  
   > epochs: 7  
   > optim_g: {type: Adam, lr: 1e-4, betas: [0.9, 0.99]}  

'''

<br/><br/>

## Code Descriptions
1. DACON_AISR_TRIAL
- EDSR, SRGAN, SWINIR


2. DACON_AISR_BEST
- RRDB, RRDB+(Self-ensemble)

