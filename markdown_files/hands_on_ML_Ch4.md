---
layout: post
title: 모델훈련
---

# 모델 훈련

## 선형 모델

선형 회귀 모델 훈련을 시키는 두가지 방법
- 직접 계산 공식을 사용하여 가장 잘 맞는 모델 파라미터를 해석적으로 구한다.
- 경사하강법을 사용하여 비용함수를 훈련 세트에 대해 최소화 시킨다. (배치 경사 하강법, 미니배치 경사 하강법, 확률적 경사 하강법)

## 비선형 모델

다항 회귀의 경우에는 선형 회귀보다 파라미터가 많아 과대적합(overfitting)되기 쉽다.

과대적합을 감지하고 막는 방법
- 학습 곡선(learning curve)를 사용해서 모델이 과대적합(overfitting)되는지 감지한다.
- 규제를 통해서 과대적합(overfitting)을 감소시킨다.

## 선형 회귀

선형 모델 식

$$\hat{y} = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + \cdots + \theta_{n}x_{n}$$
- $\hat{y}$ : 예측값
- n : 특성의 수
- $x_{i}$ : $i$번째 특성
- $\theta_{j}$ : j 번째 모델 파라미터
 
 이를 벡터 형식으로 나타내면

$$\hat{y} = h_{\theta}(\textbf{x}) = \bm{\theta} \cdot \textbf{x}$$

-	$\textbf{x}$ : 특성 벡터
-	$h_{\bm{\theta}}$ : 모델 파라미터 $\bm{\theta}$ 를 사용한 가설 함수

MSE cost function(loss function)
- $MSE(\textbf{X}, h_{\theta}) = \frac{1}{m} \sum_{i=1}^{m}(\bm{\theta}^{T}\textbf{x}^{(i)} - y^{(i)})^2$

>cost function : 
>우리는 cost function이 최소가 되는 $\bm{\theta}$ 를 찾는 것이 목표이다.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU2MDUxMDg2Niw5MjgxODE0NzgsMjA2OD
M1NDc2LC01Mzg0MTgyNDIsLTE3NTQ2Mjk3NDRdfQ==
-->