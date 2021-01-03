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

- 학습률이 너무 작으면 최적점에 도달하는 시간이 길어진다.
- 학습률이 너무 크면 최적점에서 점점더 멀어져 발산한다.
- 적절한 학습률을 찾는 방법으로 그리드 탐색을 사용하면 된다. 단, 수렴하는데 너무 오래걸리는 모델을 막기 위해 반복 횟수를 제한해야하는데 이때 사용하는 것이 **허용오차(tolerance)** 이다.
- 벡터의 노름이 어떤 값 $\varepsilon$보다 작아지면 알고리즘을 종료한다.

## 확률적 경사 하강업(Stochastic Gradient Descent)

확률적 경사 하강버은 매 스텝에서 한 개의 샘플을 무작위로 선택하고 그 샘플에 대한 그레디언트를 계산한다.

**SGD의 장점**
- 알고리즘의 속도가 매우 빠름
- 매우 큰 훈련 세트도 훈련시킬 수 있음
- 비용한수가 불규칙할 때 지역 최솟값에서 탈출시켜줌

**SGD의 단점**
- 배치 경사 하강법보다 불안정하다
-- 최솟값에는 근접하겠지만 아착하지는 못함
- 비슷한 맥락으로 좋은 파라미터가 구해지겠지만 최적값은 아님
- 지역 최소는 탈출하지만 전역 최솟값에 도달하기 힘듦
-- 해결하는 방법으로 담금질 기법(simulated annealing)과 유사한 방법으로 학습률을 점진적으로 감소시키는 방법이 있다.
-- 매 반복에서 학습률을 결정하는 함수를 **학습 스케쥴(learning schedule)** 이라고 한다.
- 샘플을 무작위로 선택하기에 어떤 샘플은 한 에포크에서 여러번 선택될 수도, 한번도 선택되지 않을 수도 있다.
-- 해결방법 : 에포크마다 훈련세트를 섞은 후 순서대로 하나씩 선택한다.(단, 보통 더 늦게 수렴한다는 단점이 있다)

**주의사항**
- SGD를 사용할 때 훈련 샘플이 IID를 만족해야 평균적으로 파라미터가 전역 최적점을 향해 진행한다고 보장할 수 있다.
- 이를 만드는 법은 훈련하는 동안 샘플을 섞는 것이다.

## 미니배치 경사하강법(Mini-batch Gradient Descent)

**미니배치 경사하강법이란**

- 미니배치라 부르는 임의의 작은 샘플 세트에 대해 그레디언트를 계산한다.
- 주요 장점으로는 행렬 연산에 최적화된 하드웨어(GPU)를 사용해서 얻는 성능향상이다.

미니배치를 크게하면 SGD보다 덜 불규칙하게 움직이며 SGD보다 최솟값에 더 가까이 도달하게 된다. 하지만 지역 최솟값에서 빠져나오기는 더 힘들지도 모른다.


|알고리즘  | m이 클 때 | 외부 메모리 학습지원 | n이 클때 | 하이퍼 파라미터 수 | 스케일 조정 필요 | 사이킷런|
|--|--|--|--|--|--|--|
|정규방정식  | 빠름 | No|느림|0|No|N/A|
|SVD  | 빠름 | No|느림|0|No|LinearRegression|
|배치 경사 하강법  | 느림 | No|빠름|2|Yes|SGDRegressor|
|정규방정식  | 빠름 | No|빠름|>= 2|Yes|SGDRegressor|
|정규방정식  | 빠름 | No|빠름|>= 2|Yes|SGDRegressor|

## 다항 회귀
**다항 회귀란**

- 각 특성의 거듭제곱을 새로운 특성으로 추가하고, 확장된 특성을 포함한 데이터셋에 선형 모델을 훈련시키는 기법
> sklearn의 PolynomialFeatures 사용

**PolynomialFeatures**
- sklearn의 preprocessing 클래스 중 하나로 특성이 n개인 배열을 $\frac{(n + d)!}{d!n!}$개인 배열로 변환한다.
(ex. 2 degree일 때, $(a, b) \rightarrow (1, a, b, a^2, ab, b^2)$)
- include_bias의 기본값은 True 인데 True이면 편향을 위한 특성($x_0$)인 1이 추가된다.

PolynomialFeatures로 데이터를 전처리한 후 LinearRegression 을 통해서 회귀를 적용하면 된다.

## 학습 곡선

우리가 다항회귀를 할 경우 과연 얼마나 복잡한 모델을 사용하는 것이 가장 효과적인가에 대한 의문점이 생길 수 있다.

어떤 데이터가 2차 방정식으로 생성되었다고 하자. 선형회귀를 할 경우 과소적합이고 300차 방정식으로 회귀 모델을 잡을 경우 과대적합이 일어나게 된다.

이를 해결하는 방법으로는 교차 검증과 학습 곡선을 살펴보는 두 가지 방법이 존재한다.
1. 교차  검증
	-  훈련 데이터에서 성능이 좋지만 교차 검증 점수가 나쁘다면 모델이 과대적합.
	- 양쪽에 모두 좋지 않으면 과소적합

2. 학습 곡선
	- 훈련 세트와 검증 세트의 모델 성능을 훈련 세트 크기의 함수로 나타낸 그래프
	- 이 그래프를 생성하는 방법은 훈련 세트에서 크기가 다른 서브 세트를 만들어 모델을 여러 번 훈련시키면 된다.

![enter image description here](data:image/png;base64,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)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQ5ODkzNTgyOSw3NDkxNTMyODcsMTQyMj
A1ODM0NiwtNjE5NDIyNTg0LDU5Njg2OTM5OCwtNDM5MTU3Njc3
LC04MzM3NTAxNTMsLTYzNDI2MTk3MCw4NTEwMTM2MSwtMTg2MT
IyMzk4NywtMTg2MTIyMzk4NywtMTYwMDUzMzc2OV19
-->