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

### 선형 모델 식

$$\hat{y} = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + \cdots + \theta_{n}x_{n}$$
- $\hat{y}$ : 예측값
- n : 특성의 수
- $x_{i}$ : $i$번째 특성
- $\theta_{j}$ : j 번째 모델 파라미터
 
 이를 벡터 형식으로 나타내면

$$\hat{y} = h_{\theta}(\textbf{x}) = \bm{\theta} \cdot \textbf{x}$$

-	$\textbf{x}$ : 특성 벡터
-	$h_{\bm{\theta}}$ : 모델 파라미터 $\bm{\theta}$ 를 사용한 가설 함수

MSE cost function은 아래와 같이 표현된다.
- $MSE(\textbf{X}, h_{\theta}) = \frac{1}{m} \sum_{i=1}^{m}(\bm{\theta}^{T}\textbf{x}^{(i)} - y^{(i)})^2$

>cost function : 예측 값과 실제 값의 차이를 나타낸 함수(미묘한 차이가 있긴하지만 loss function이라고 생각해도 무관하다.)
>우리는 cost function이 최소가 되는 $\bm{\theta}$ 를 찾는 것이 목표이다.

### 정규방정식

$\bm{\theta}$를 구하기 위한 해석적인 방법이 있는데 이 방정식을 정규방정식이라 부른다.

$$\hat{\bm{\theta}} = (\textbf{X}^{T}\textbf{X})^{-1}\textbf{X}^T\textbf{y}$$
- $\hat{\bm{\theta}}$ : 비용함수를 최소하하는 ${\bm{\theta}}$
- $\textbf{y}$ : 타깃 벡터

$$\hat{\textbf{y}} = \textbf{X} \bm{\theta}  $$

- $\hat{\textbf{y}}$ : 예측 벡터

위와 같은 방법으로 우리는 $\hat\bm{\theta}$를 구할 수 있고 이를 통해 $\hat{\textbf{y}}$를 예측할 수 있다.

### 사이킷런에서의 선형회귀

LinearRegression 클래스를 사용하면 매우 쉽게 $\hat\bm{\theta}$를 구할 수 있다.

LinearRegression 클래스는 scipy.linalg.lstsq() 함수를 기반으로 작동한다.

이 함수는 $\hat\bm{\theta} = \textbf{X}^+\textbf{y}$를 계산한다.

$\textbf{X}^+$는 유사역행렬이다.
> 유사역행렬은 특잇값 분해라는 표준 행렬 분해 기법을 사용해 계산된다.
> 
> 특잇값분해(SVD) : $\textbf{X} = \textbf{U}\Sigma\textbf{V}^T$로 분해, 유사역행렬의 경우 $\textbf{X}^+ = \textbf{U}\Sigma^+\textbf{V}^T$로 분해한다.
> 
> $\Sigma^+$ 는 $\Sigma$를 먼저 구한후 임곗값보다 작은 모든 수를 0으로 바꾼다.

유사역행렬의 장점으로는 $\textbf{X}^{T}\textbf{X}$의 역행렬은 존재하지 않는 경우가 있지만 유사역행렬은 항상 존재한다.

---
위와 같이 정규방정식 혹은 다른 알고리즘을 이용해서 선형 회귀 모델을 예측하는 것은 매우 빠르다는 장점을 갖고 있다. 

단, 정규방정식과 SVD방식 모두 특성 수가 많아 지면 매우 느려진다.

또한 예측 계산 복잡도는 샘플 수와 특성 수에 선형적이다.

## 경사하강법(Gradient Descent)

### 경사하강법이란?
 비용 함수를 최적화하기 위해 반복해서 파라미터를 조정해가는 것
 
>$\bm{\theta}$에 대해 비용 함수의 현재 gradient를 계산하고 이를 감소하는 방향으로 진행한다. 그리고 gradient가 0이 되며 최솟값에 도달한 것이다.

$\bm{\theta}$를 임의의 값으로 random initialization 후 최솟값에 수렴할 때까지 점진적으로 진행한다.

### 학습률(learning rate)

경사하강법에서 중요한 파라미터로 **학습률 (learning rate)** 이 있다. 학습률은 하이퍼 파라미터로 스텝의 크기이다.

>학습률이 너무 작으면 수렴하기 위해 반복을 많이 진행해야 한다.

###  주의사항

경사하강법을 사용할 때는 반드시 모든 특성이 같은 스케일을 갖도록 만들어야한다. (StandardScaler)

## 배치 경사 하강법(Batch Gradient Descent)

배치 경사하강법을 구현하려면 각 모델 파라미터 $\theta_j$에 대해 비용 함수의 그레디언트를 계산해야한다.

**비용함수의 편도함수**
$$\frac{\partial }{\partial \bm{\theta} _{j}} MSE(\bm\theta) = \frac{2}{m}\sum_{i=1}^{m}(\bm\theta^{T}\textbf{x}^{(i)} - y^{(i)})x{_{j}}^{(i)}$$

> $MSE(\textbf{X}, h_{\theta}) = \frac{1}{m} \sum_{i=1}^{m}(\bm{\theta}^{T}\textbf{x}^{(i)} - y^{(i)})^2$ 를 $\theta_j로$ 편미분한 것

일일이 편도함수를 계산하는 대신에 한꺼번에 계산할 수 있는 방법이 있다.

**비용함수의 그레디언트 벡터**

$$\nabla_\theta MSE(\boldsymbol{\theta}) = \begin{pmatrix}
\ \frac{\partial }{\partial \boldsymbol{\theta} _{0}} MSE(\boldsymbol{\theta})
\\ \frac{\partial }{\partial \boldsymbol{\theta} _{1}} MSE(\boldsymbol{\theta})
\\ \vdots 
\\ \frac{\partial }{\partial \boldsymbol{\theta} _{n}} MSE(\boldsymbol{\theta})
\end{pmatrix}
= \frac{2}{m}\boldsymbol{X}^T(\boldsymbol{X\theta - }\textbf{y})$$
>이 부분에 대해서는 조금 더 공부가 필요하다. 식의 이해가 부족함


위의 방법으로 그레디언트 벡터가 구해지면 반대 방향으로 가야한다.

즉, $\boldsymbol{\theta}^{(next\_step)} = \boldsymbol\theta - \eta\nabla_\theta MSE(\boldsymbol{\theta})$
>$\eta$는 learning rate이다.

학습률이 너무 작으면 최적점에 도달하는 시간이 길어진다.
학습률이 너무 크면 최적점에서 점점더 멀어져 발산한다.


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5NjU0MTk5MjAsODUxMDEzNjEsLTE4Nj
EyMjM5ODcsLTE4NjEyMjM5ODcsLTE2MDA1MzM3NjldfQ==
-->