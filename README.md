# 🚘 자율주행 센서의 안테나 성능 예측 AI 경진대회 🚘

<img src="https://user-images.githubusercontent.com/55012723/210489354-d39644e2-f938-4b35-97aa-8b4338a73a5c.jpg"  width="500" height="500"/>

Regression을 통한 예측 정확도를 극대화해보자!

<br/><br/>

## Team

**Name** : 안장최

**Members** : 고려대학교 산업경영공학부 안영지, 장수혁, 최대원

<br/><br/>

## Project Descriptions

**Link** : [자율주행 센서의 안테나 성능 예측 AI 경진대회 (DACON)](https://dacon.io/competitions/official/235927/overview/description)


**[목표]**  
공정 데이터를 활용하여 Radar 센서의 안테나 성능 예측을 위한 AI 모델 개발  <br/>

공정 데이터와 제품 성능간 상관 분석을 통해 제품의 불량을 예측/분석


**[배경]**  
Radar는 자율주행 차에 있어 차량과의 거리, 상대 속도, 방향 등을 측정해주는 필수적인 센서 부품 <br/>
LG에서는 제품의 성능 평가 공정에서 양품과 불량을 선별 중. <br/>
AI 기술을 활용하여 공정 데이터와 제품 성능간 상관 분석을 통해 제품의 불량을 예측/분석하고, <br/>
수율을 극대화하여 불량으로 인한 제품 폐기 비용을 감축시키는 것이 목표.


**[데이터]**

-Train Data : 39607개 (수치형 정형 데이터)

-Test Data : 39608개

-X 변수 : 56개 (공정 순서대로 번호 라벨링)
 > 
    - PCB 체결 시 단계별 누름량(1~4)
    - 방열(TIM) 재료 면적(1~3)
    - 방열(TIM) 재료 무게(1~3)
    - 커넥터 위치 기준 좌표 → 장비의 원점으로부터 거리 값
    - 각 안테나 패드 위치(높이) 차이
    - 안테나 패드 위치
    - 1st 스크류 삽입 깊이(1~4)
    - 커넥터 핀 치수(1~6)
    - 2nd 스크류 삽입 깊이(1~4) 
    - 스크류 체결 시 분당 회전수 (1~4)
    - 하우징 PCB 안착부 치수(1~3)
    - 레이돔 치수 (안테나 1~4번 부위)
    - 안테나 부분 레이돔 기울기
    - 실란트 본드 소요량
    - Cal 투입 전 대기 시간(Calibration 공정 -> RF 성능 편차 없도록 조정해주는 과정)
    - RF 1~7 부분의 SMT 납 량

-Y 변수 : 14개
 > <img width="300" alt="image" src="https://user-images.githubusercontent.com/55012723/210501843-15250981-b4c3-4567-9889-4f31bfcfb540.png">



**[주최]**  
LG AI Research 


**[평가 산식]**  
Normalized RMSE (NRMSE)

```python

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(1,15): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    return score
    
```

<br/><br/>

## 👣 Score
Public : 1.94045(122th) <br/>
Private : 1.96222(126th)

<br/><br/>

## 🌐 Environment

Google Colab
OS: macOS Ventura 13.0 <br/>
CPU: Intel(R) Xeon(R) CPU @ 2.20GHz 

<br/><br/>

## 🔥 Competition Strategies

**1. Feature Enginnering **  
- Adding Average SMT column
- Deleting Validation columns

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

