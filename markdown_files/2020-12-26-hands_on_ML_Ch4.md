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

$$\hat{y} = h_{\theta}(\textbf{x}) = \boldsymbol{\theta} \cdot \textbf{x}$$

-	$\textbf{x}$ : 특성 벡터
-	$h_{\boldsymbol{\theta}}$ : 모델 파라미터 $\boldsymbol{\theta}$ 를 사용한 가설 함수

MSE cost function은 아래와 같이 표현된다.
- $MSE(\textbf{X}, h_{\theta}) = \frac{1}{m} \sum_{i=1}^{m}(\boldsymbol{\theta}^{T}\textbf{x}^{(i)} - y^{(i)})^2$

>cost function : 예측 값과 실제 값의 차이를 나타낸 함수(미묘한 차이가 있긴하지만 loss function이라고 생각해도 무관하다.)
>우리는 cost function이 최소가 되는 $\boldsymbol{\theta}$ 를 찾는 것이 목표이다.

### 정규방정식

$\boldsymbol{\theta}$를 구하기 위한 해석적인 방법이 있는데 이 방정식을 정규방정식이라 부른다.

$$\hat{\boldsymbol{\theta}} = (\textbf{X}^{T}\textbf{X})^{-1}\textbf{X}^T\textbf{y}$$
- $\hat{\boldsymbol{\theta}}$ : 비용함수를 최소하하는 ${\boldsymbol{\theta}}$
- $\textbf{y}$ : 타깃 벡터

$$\hat{\textbf{y}} = \textbf{X} \boldsymbol{\theta}  $$

- $\hat{\textbf{y}}$ : 예측 벡터

위와 같은 방법으로 우리는 $\hat\boldsymbol{\theta}$를 구할 수 있고 이를 통해 $\hat{\textbf{y}}$를 예측할 수 있다.

### 사이킷런에서의 선형회귀

LinearRegression 클래스를 사용하면 매우 쉽게 $\hat\boldsymbol{\theta}$를 구할 수 있다.

LinearRegression 클래스는 scipy.linalg.lstsq() 함수를 기반으로 작동한다.

이 함수는 $\hat\boldsymbol{\theta} = \textbf{X}^+\textbf{y}$를 계산한다.

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
 
>$\boldsymbol{\theta}$에 대해 비용 함수의 현재 gradient를 계산하고 이를 감소하는 방향으로 진행한다. 그리고 gradient가 0이 되며 최솟값에 도달한 것이다.

$\boldsymbol{\theta}$를 임의의 값으로 random initialization 후 최솟값에 수렴할 때까지 점진적으로 진행한다.

### 학습률(learning rate)

경사하강법에서 중요한 파라미터로 **학습률 (learning rate)** 이 있다. 학습률은 하이퍼 파라미터로 스텝의 크기이다.

>학습률이 너무 작으면 수렴하기 위해 반복을 많이 진행해야 한다.

###  주의사항

경사하강법을 사용할 때는 반드시 모든 특성이 같은 스케일을 갖도록 만들어야한다. (StandardScaler)

## 배치 경사 하강법(Batch Gradient Descent)

배치 경사하강법을 구현하려면 각 모델 파라미터 $\theta_j$에 대해 비용 함수의 그레디언트를 계산해야한다.

**비용함수의 편도함수**
$$\frac{\partial }{\partial \boldsymbol{\theta} _{j}} MSE(\boldsymbol\theta) = \frac{2}{m}\sum_{i=1}^{m}(\boldsymbol\theta^{T}\textbf{x}^{(i)} - y^{(i)})x{_{j}}^{(i)}$$

> $MSE(\textbf{X}, h_{\theta}) = \frac{1}{m} \sum_{i=1}^{m}(\boldsymbol{\theta}^{T}\textbf{x}^{(i)} - y^{(i)})^2$ 를 $\theta_j로$ 편미분한 것

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

**학습 곡선**

---
![linear learning curve](https://2bhapby.github.io/images/learning_curve_linear.png)

위 그래프는 선형 회귀 모델의 학습 곡선이다.

- 훈련 데이터는 0에서 시작해서 훈려 세트에 한 개 혹은 두 개의 샘플이 있을때 완벽하게 작동한다. 하지만 그 이후에는 데이터가 비선형이기 때문에 어느 정도 평평해질 때까지 오차가 상승한다. 어느 정도 평평해진 이후에는 데이터가 추가되어도 크게 변동이 없다.

- 검증 데이터는 훈련 샘플 수가 적을 때는 훈련이 제대로 될 수 없어 오차가 매우크다. 그 이후 훈련 샘플이 추가됨에 따라 검증 오차가 천천히 감소한다. 하지만 데이터를 잘 모델링 할 수 없으므로 훈련 세트 그래프와 가까워진다.
- 과소적합의 전형적인 모습이다. 
- 두 곡선이 수평한 구간을 만들고 꽤 높은 오차에서 매우 가까이 근접

---
![10 degree learning curve](https://2bhapby.github.io/images/learning_curve_10deg.png)

위 그래프는 10차 다항식으로 회귀한 것이다.
- 훈련 데이터의 오차가 선형 회귀 모데로다 훨씬 낮다

- 두 곡선 사이에 공간이 있다. 즉, 훈련데이터에서의 성능이 검증 데이터에서보다 훨씬 낫다.
- 과대적합의 전형적인 모습이다.
- 더 큰 훈련 세트를 사용하면 두 곡선이 점점 가까워진다.
---
**편향/분산 트레이드오프**

- **편향**
	--일반화 오차 중에서 편향은 잘못된 가정으로 인한 것
	(ex. 데이터는 실제로 2차지만 선형으로 가정)
	--편향이 큰 모델은 훈련 데이터에 과소적합되기 쉬움

- **분산**
	-- 훈련 데이터에 있는 작은 변동에 모델이 과도하게 민감하기 때문에 나타난다.
	-- 자유도가 높은 모델이 높은 분산을 갖기 쉬워 과대적합되는 경향이 있다.

- **줄일 수 없는 오차(irreducible error)**
	-- 데이터 자체에 있는 잡음 때문에 발생
	--유일한 해결 방법은 데이터에서 잡음 제거

## 규제가 있는 선형 모델
> 규제는 과대적합을 감소시키는 좋은 방법이다.

> 선형회귀 모델에서는 모델의 가중치를 제한함으로써 규제를 가한다. (릿지 회귀, 라쏘 회귀, 엘라스틱넷)

> 입력 특성의 스케일에 민감하기에 스케일을 맞춰주는 것이 중요하다.

---

### 릿지 회귀(Ridge)

릿지 회귀는 비용 함수에 규제(항)이 추가된 선형 회귀 버전이다.
>**규제항**: $\alpha \sum_{i=1}^{n} \theta_i^2$

모델의 훈련이 끝나면 모델의 성능을 규제가 없는 성능 지표로 평가한다.

**비용함수**
$$\boldsymbol{J(\theta)} = MSE(\boldsymbol{\boldsymbol\theta}) + \alpha \frac{1}{2}\sum_{n}^{i=1}\theta_i^2$$
>$\textbf{w}$ 를 특성의 가중치 벡터라고 정의하면 규제항은 $\frac12(\left \| \textbf{w}\right \|_2)^2$와 같다.

이때 편향 $\theta_0$는 규제되지 않는다.

**릿지 회귀의 정규방정식**

$\hat\theta = (\textbf{X}^T\textbf{X} + \alpha\textbf A)^{-1}\textbf X^T\textbf y$

---

### 라쏘 회귀(Least Absolute Shrinkage and Selection Operator)

규제항으로 $l_2$ norm 대신에 $l_1$ norm을 사용한다.

**비용함수**
$$\boldsymbol{J(\theta)} = MSE(\boldsymbol{\boldsymbol\theta}) + \alpha \frac{1}{2}\sum_{n}^{i=1}|\theta_i|$$

- **특징**
-- 덜 중요한 특성의 가중치를 제거하려고 한다.($\theta_i$를 0으로 만든다.)
-- 자동으로 특성 선택을 하고 희소 모델을 만든다.
--$\theta_i = 0$에서 미분 불가능하다. 하지만 이때 서브그레디언트 벡터 $\textbf g$를 사용하면 경사하강법을 적용하는데 문제 없다.



**서브그레디언트 벡터 g**

$$g(\boldsymbol \theta, J) = \nabla_\theta MSE(\boldsymbol\theta) + \alpha \begin{pmatrix} sign(\theta_1)
\\ sign(\theta_2) 
\\ \vdots 
\\ sign(\theta_n)\end{pmatrix}$$ 이때 $sign(\theta_i) = -1, 0, 1$(각각 0보다 작을 때, 0일 때, 0보다 클 때)


---

**릿지 회귀 라쏘 회귀의 가장 큰 차이점**

릿지 회귀는 특성값을 0으로 수렴하게 할 뿐 0이 되진 않는다. 하지만 라쏘 회귀는 0으로 만들어버린다.

그 이유는 그래프로 생각해보면 간단하다. 특성이 2개라고 생각을 해보자. $l_1$ 규제의 경우 등고선을 그려보면 마름모 꼴이고 $l_2$ 규제의 경우 원형이다.

등고선과 수직으로 줄어든다고 보면 마름모 꼴의 경우 어느 한 파라미터가 먼저 0에 도달한다. 그렇기에 어떠한 특성은 0이 되어버린다.

하지만 원형의 경우 둘 모두 조금씩 줄다가 동시에 원점에서 만나기 때문에 0에 수렴할 뿐 0이 되지 않는다.

또한 릿지회귀는 전역 최적점에 가까워질수록 그레디언트가 작아진다.

---

### 엘라스틱넷
>릿지 라쏘 회귀의 절충 모델

**비용함수**
$$\boldsymbol{J(\theta)} = MSE(\boldsymbol{\boldsymbol\theta}) + r\alpha \frac{1}{2}\sum_{n}^{i=1}|\theta_i| + \frac{1-r}{2}\alpha \frac{1}{2}\sum_{n}^{i=1}\theta_i^2$$

- 대부분의 경우 : 릿지
- 특성이 몇개뿐이라고 의심 : 라쏘, 엘라스틱넷
- 특성수가 훈련 샘플 수보다 많거나 특성 몇개가 강하게 연관 : 엘라스틱넷

---

### 조기종료

에러가 최솟값에 도달하면 훈련을 바로 중지시키는 것

>SGD 혹은 미니배치 경사하강법은 곡선이 매끄럽지 않아 검증 에러가 일정 시간 동안 최솟값보다 클 때 학습을 멈추고 검증에러가 최소였을 때의 모델 파라미터로 되돌린다.


## 로지스틱 회귀

샘플이 특정 클래스에 속할 확률을 추정하는 데 널리 사용한다. 즉, 회귀 알고리즘이지만 분류에서도 사용한다.

추정 확률이 50%가 넘으면 그 샘플이 해당 클래스에 속한다고 예측한다. 아니면 속하지 않는다고 예측한다. (레이블 1인 양성클래스, 레이블 0인 음성 클래스)

### 확률 추정

입력 특성의 가중치의 합을 계산한다.(편향 더함) 대신 선형 회귀처럼 바로 결과를 출력하지 않고 결괏값의 로지스틱을 출력한다.


$\hat{p}= h_\theta(\textbf{x}) = \sigma(\boldsymbol{\theta}^T\textbf{x})$
> $\sigma$는 sigmoid 함수로 $\sigma(t) = \frac{1}{1+exp(-t)}$


**로지스틱 회귀 모델 예측**

$\hat{y} = 
\left\{\begin{matrix} 0  \quad \hat{p} < 0.5 일때 
\\ 1  \quad \hat{p} \geq 0.5 일때 
\end{matrix}\right.$

$t < 0$ 이면 $\sigma(t) < 0.5$이고 $t \geq 0$ 이면 $\sigma(t) \geq 0.5$ 이므로 로지스틱 회귀 모델은 $\boldsymbol{\theta}^T\textbf{x}$가 양수일 때 1, 음수일 때 0이라고 예측

>t = logit = log-odds
>$t = logit(p) = log(p / ( 1 - p ))$

---

### 로지스틱 회귀 모델 훈련과 비용함수

**훈련 목적**

- 양성 샘플$(y=1)$에 대해서는 높은 확률을 추정, 음성 샘플$(y=0)$에 대해서는 낮은 확률을 추정하는 모델의 파라미터 벡터 $\boldsymbol\theta$를 찾는 것


**비용 함수**

- 하나의 훈련샘플 $\textbf{x}$에 대한 비용함수

$$c(\boldsymbol{\theta}) = \left\{\begin{matrix}
-log(\hat{p}) \quad\quad\quad y = 1 일때 \\ -log(1-\hat{p}) \quad y = 0 일때
\end{matrix}\right.$$

> t가 0에 가까워지면 -log(t)가 매우 커진다 -> 양성 샘플에 대하여 비용이 크게 증가하므로 타당하다.

> t가 1에 가까워지면 -log(t) 가 0이 된다 -> 양성 샘플에 대하여 비용이 0에 가까워지므로 타당하다.

>  음성 샘플에 대해서도 유사하게 적용가능하다.

- 로지스틱 회귀의 비용 함수(로그 손실)

$$\textbf{J}(\boldsymbol{\theta}) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}log(\hat{p}^{(i)})+(1-y^{(i)})log(1-\hat{p}^{(i)})]$$

최솟값을 계산하는 알려진 해는 없지만 볼록 함수이므로 경사 하강법이 전역 최솟값을 찾는 것을 보장한다.

- 로지스틱 비용 함수의 편도함수

$$\frac{\partial }{\partial\theta_j}J(\boldsymbol{\theta}) = \frac{1}{m}\sum_{i=1}^{m}(\sigma(\boldsymbol{\theta}^T\textbf{x}^{(i)}) - y^{(i)})x_j^{(i)}$$

### 결정 경계

로지스틱 회귀에서 양쪽의 확률이 똑같이 50%가 되는 곳 근방에서 결정 경계가 만들어진다.


### 소프트맥스 회귀

로지스틱 회귀 모델은 여러 개의 이진 분류기를 훈련시켜 연결하지 않고 직접 다중 클래스를 지원하도록 일반화될 수 있다. 이를 **소프트맥스 회귀** 혹은 **다항 로지스틱 회귀**라고 한다.

**개념**

- 샘플 $\textbf x$가 주어지면 소프트맥스 회귀 모델이 각 클래스 $k$에 대한 점수 $s_k(\textbf x)$를 계산
- 그 점수에 소프트맥스 함수를 적용하여 각 클래스의 확률을 추정한다.

>클래스 k에 대한 소프트맥스 점수
> $s_k(\textbf{x}) = (\boldsymbol{\theta}^{(k)})^T\textbf{x}$

각 클래스는 자신만의 파라미터 벡터 $\boldsymbol{\theta}^{(k)}$가 있다. 이 벡터들은 파라미터 행렬 $\Theta$에 행으로 저장된다.

>소프트맥스 함수
> $\hat{p}_k = \sigma(\textbf{s}(\textbf x))_k = \frac{exp(s_k(\mathbf{x}))}{\sum_{j=1}^{K}exp(s_j(\mathbf{x}))}$
> - K : 클래스 수
>  - $\textbf{s}(\textbf x)$ : 샘플 $\textbf x$에 대한 각 클래스의 점수를 담은 벡터
>  - $\sigma(\textbf{s}(\textbf{x}))_k$ : 샘플  $\textbf{x}$에 대한 각 클래스의 점수가 주어졌을 때 이 샘플이 클래스 k에 속할 추정 확률


$\hat{y}  = \underset{k}{argmax}\sigma(\textbf{s}(\textbf{x}))_k = \underset{k}{argmax}s_k((\textbf x)) = \underset{k}{argmax}((\boldsymbol{\theta}^{(k)})^T\textbf x)$
 >argmax 는 함수를 최대화하는 변수의 값을 반환

*소프트맥스 회귀 분류기는 한 번에 하나의 클래스만 예측한다. 그래서 상호 배타적인 클래스에서만 사용해야한다.*

---

**훈련방법**

>목표: 모델이 타깃 클래스에 대해서 높은 확률을 추정하도록 만드는 것

크로스 엔트로피 비용함수를 최소화하는 것은 타깃 클래스에 대해 낮은 확률을예측하는 모델을 억제하므로 이 목적에 부합

추정된 클래스의 확률이 타깃 클래스에 얼마나 잘 맞는지 측정하는 용도로 종종 사용됨

- 크로스 엔트로피 비용 함수

$J(\Theta) = - \frac{1}{m} \sum_{i=1}^{m}\sum_{k=1}^{K}y_k^{(i)}log(\hat{p}_k^{(i)})$

>$y_k^{(i)}$는 $i$ 번째 샘플이 클래스 $k$에 속할 타깃 확률
>샘플이 클래스에 속하는지 아닌지에 따라 1 또는 0

- 클래스 k에 대한 크로스 엔트로피의 그레디언트 벡터

	$\nabla_{\boldsymbol{\theta}^{(k)}}\textbf J(\boldsymbol{\Theta})= \frac{1}{m}\sum_{i=1}^{m}(\hat p_k^{(i)} - y_k^{(i)})\textbf{x}^{(i)}$


sklearn의 LogisticRegression은 클래스가 둘 이상일 때 기본적으로 OvA 전략을 사용
multi_class 매개변수를 "multinomial"로 바꾸면 소프트맥스 회귀를 사용할 수 있음
solver 매개변수에 "lbfgs"와 같이 소프트맥스 회귀를 지원하는 알고리즘을 지정해야 함



reference: Hands-on Machine Learning with Scikit-learn, Keras & Tensorflow, Second Ed.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1OTI3OTcxODUsLTEzNzkzMzM2OTcsLT
EwNzQxNDkyNjcsLTMyNDg5NTA2OSw1OTQ5NDE0MjUsLTU1ODAz
ODY0MywtMTA5NDY1Njk0NiwxMDgzMDQ5OTIyLC0xNTM1OTExND
M4LC05OTYxNzE2OTYsMjM0MzU2ODA2LDEwMTEzNDY2NjksLTIw
OTQ1MzUwNTUsLTMwNzMxNDQ3MSwtMTkwMjU5MjU3NCwxMjUxMD
UwNDg4LC02MzEzMTM3MjYsLTEyNjE2MTYxMywyMDUzNTAzODQ5
LC0xNDQ3ODY2Njk2XX0=
-->