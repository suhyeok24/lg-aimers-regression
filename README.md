# π μμ¨μ£Όν μΌμμ μνλ μ±λ₯ μμΈ‘ AI κ²½μ§λν π

<img src="https://user-images.githubusercontent.com/55012723/210489354-d39644e2-f938-4b35-97aa-8b4338a73a5c.jpg"  width="500" height="500"/>

Regressionμ ν΅ν μμΈ‘ μ νλλ₯Ό κ·Ήλνν΄λ³΄μ!

<br/><br/>

## Team

**Name** : μμ₯μ΅

**Members** : κ³ λ €λνκ΅ μ°μκ²½μκ³΅νλΆ μμμ§, μ₯μν, μ΅λμ

<br/><br/>

## Project Descriptions

**Link** : [μμ¨μ£Όν μΌμμ μνλ μ±λ₯ μμΈ‘ AI κ²½μ§λν (DACON)](https://dacon.io/competitions/official/235927/overview/description)


**[λͺ©ν]**  
κ³΅μ  λ°μ΄ν°λ₯Ό νμ©νμ¬ Radar μΌμμ μνλ μ±λ₯ μμΈ‘μ μν AI λͺ¨λΈ κ°λ°  <br/>

κ³΅μ  λ°μ΄ν°μ μ ν μ±λ₯κ° μκ΄ λΆμμ ν΅ν΄ μ νμ λΆλμ μμΈ‘/λΆμ


**[λ°°κ²½]**  
Radarλ μμ¨μ£Όν μ°¨μ μμ΄ μ°¨λκ³Όμ κ±°λ¦¬, μλ μλ, λ°©ν₯ λ±μ μΈ‘μ ν΄μ£Όλ νμμ μΈ μΌμ λΆν <br/>
LGμμλ μ νμ μ±λ₯ νκ° κ³΅μ μμ μνκ³Ό λΆλμ μ λ³ μ€. <br/>
AI κΈ°μ μ νμ©νμ¬ κ³΅μ  λ°μ΄ν°μ μ ν μ±λ₯κ° μκ΄ λΆμμ ν΅ν΄ μ νμ λΆλμ μμΈ‘/λΆμνκ³ , <br/>
μμ¨μ κ·Ήλννμ¬ λΆλμΌλ‘ μΈν μ ν νκΈ° λΉμ©μ κ°μΆμν€λ κ²μ΄ λͺ©ν.


**[λ°μ΄ν°]**

-Train Data : 39607κ° (μμΉν μ ν λ°μ΄ν°)

-Test Data : 39608κ°

-X λ³μ : 56κ° (κ³΅μ  μμλλ‘ λ²νΈ λΌλ²¨λ§)
 > 
    - PCB μ²΄κ²° μ λ¨κ³λ³ λλ¦λ(1~4)
    - λ°©μ΄(TIM) μ¬λ£ λ©΄μ (1~3)
    - λ°©μ΄(TIM) μ¬λ£ λ¬΄κ²(1~3)
    - μ»€λ₯ν° μμΉ κΈ°μ€ μ’ν β μ₯λΉμ μμ μΌλ‘λΆν° κ±°λ¦¬ κ°
    - κ° μνλ ν¨λ μμΉ(λμ΄) μ°¨μ΄
    - μνλ ν¨λ μμΉ
    - 1st μ€ν¬λ₯ μ½μ κΉμ΄(1~4)
    - μ»€λ₯ν° ν μΉμ(1~6)
    - 2nd μ€ν¬λ₯ μ½μ κΉμ΄(1~4) 
    - μ€ν¬λ₯ μ²΄κ²° μ λΆλΉ νμ μ (1~4)
    - νμ°μ§ PCB μμ°©λΆ μΉμ(1~3)
    - λ μ΄λ μΉμ (μνλ 1~4λ² λΆμ)
    - μνλ λΆλΆ λ μ΄λ κΈ°μΈκΈ°
    - μ€λνΈ λ³Έλ μμλ
    - Cal ν¬μ μ  λκΈ° μκ°(Calibration κ³΅μ  -> RF μ±λ₯ νΈμ°¨ μλλ‘ μ‘°μ ν΄μ£Όλ κ³Όμ )
    - RF 1~7 λΆλΆμ SMT λ© λ

-Y λ³μ : 14κ°
 > <img width="300" alt="image" src="https://user-images.githubusercontent.com/55012723/210501843-15250981-b4c3-4567-9889-4f31bfcfb540.png">



**[μ£Όμ΅]**  
LG AI Research 


**[νκ° μ°μ]**  
Normalized RMSE (NRMSE)

```python

def lg_nrmse(gt, preds):
    # κ° Y Featureλ³ NRMSE μ΄ν©
    # Y_01 ~ Y_08 κΉμ§ 20% κ°μ€μΉ λΆμ¬
    all_nrmse = []
    for idx in range(1,15): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    return score
    
```

<br/><br/>

## π£ Score
Public : 1.94045(122th) <br/>
Private : 1.96222(126th)

<br/><br/>

## π Environment

Google Colab
OS: macOS Ventura 13.0 <br/>
CPU: Intel(R) Xeon(R) CPU @ 2.20GHz 

<br/><br/>

## π₯ Competition Strategies

**1. Feature Enginnering**  
- Adding Average SMT column
- Deleting Validation columns : all the same value 
- Clustering By Process
  > Identify characteristics of sample data and classify into clusters

  > Cluster numbers for 4 processes are reflected as features

  > Using KMeans clustering

<br/>

**2. Modeling Methods**

- Build a model for each y variable (14 models)

- Using boosting models
  > GradientBoostingRegressor

  > XGBRegressor

  > LGBMRegressor

  > CatBoostRegressor

- Stacking
  > 1. Simply averaging the predictions for cloned model :  AveragingModels (best)

  > 2. Train the cloned meta-model using the out-of-fold predictions as new feature : StackingAveragedModels


<br/>

**3. Standardization**
- StandardScaler(Mean:0, Std:1)
  > Ignore the units of each characteristic and simply compare them by value 

 

<br/>

**4. Hyperparameter Tuning**
- Optuna : to minimize RMSE
  > Simple and fast 

  > Parallel processing is possible

  > Equipped with various optimization algorithms of the latest trend


<br/><br/> 

