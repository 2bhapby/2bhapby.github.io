---
layout: post
title: 앙상블 학습과 랜덤 포레스트
---

# 앙상블 학습과 랜덤 포레스트

wisdom of the crowd와 비슷하게 일련의 예측기로부터 예측을 수집하면 가장 좋은 모델 하나보다 더 좋은 예측을 얻을 수 있을 것이다.

일련의 예측기를 **앙상블**이라고 부른다. 그렇기 때문에 이를 **앙상블 학습(ensemble learning)** 이라고 하며 앙상블 학습 알고리즘을 **앙상블 방법(ensemble method)** 이라고 한다.

**앙상블 방법**

- 훈련 세트로부터 무작위로 각기 다른 서브셋을 만들어 일련의 결정 트리 분류기를 훈련시킨다.
- 예측은 모든 개별 트리의 예측을 구하면 된다.
- 그 후 가장 많은 선택을 받은 클래스를 예측으로 삼는다.
- 결정 트리의 앙상블을 **랜덤 포레스트** 라고 한다.
-- 가장 강력한 ML 알고리즘 중 하나
- 예로는 **배깅, 부스팅, 스태킹**이 있다.


## 투표 기반 분류기

**직접 투표 분류기**
- 여러 분류기의 예측을 모아서 가장 많이 선택된 클래스를 예측하는 것
-- 다수결 투표로 정해지는 것
- 각 분류기가 **약한 학습기** 라도 충분하게 많고 다양하면 **강한 학습기**가 될 수 있다.
- 이것이 가능한 이유는 **큰 수의 법칙(law of large numbers)** 때문이다.
- 앙상블 방법은 예측기가 가능한 서로 독립적일 때 최고의 성능을 발휘한다.

**간접 투표 분류기**
- 개별 분류기의 예측을 평균 내어 확률이 가장 높은 클래스를 예측한다.
- 확률이 높은 투표에 비중을 더 두기 때문에 직접 투표 방식 보다 성능이 높다.
- 이를 사용하기 위해서는 voting="hard"를 voting="soft" 로 바꾸고 모든 분류기가 클래스의 확률을 추정할 수 있으면 된다.

## 배깅과 페이스팅

알고리즘을 사용하고 훈련 세트의 서브셋을 무작위로 구성하여 분류기를 각기 다르게 학습시킨다.

**샘플링 방식**
- 배깅(bagging/bootstrap aggregation)
-- 훈련세트에서 중복을 허용하는 방식
-- 한 예측기를 위해 같은 훈련 샘플을 여러번 샘플링 할 수 있다.

- 페이스팅(pasting)
-- 중복을 허용하지 않는 방식

모든 예측기가 훈련을 마치면 앙상블은 모든 예측기의 예측을 모아서 새로운 샘플에 대한 예측을 만든다.
- 수집함수는 전형적인 분류일 때는 통계적 최빈값(statistical mode)을 예측한다.

- 회귀일 때는 평균을 계산해 예측한다.

개별 예측기는 원본 훈련 세트로 훈련시킨 것보다 훨씬 크게 편향되어 있지만 수집 함수를 통과하면 편향과 분산이 모두 감소한다.

### 사이킷런의 배깅과 페이스팅

**BaggingClassifier, BaggingRegressor** 
- 페이스팅은 bootstrap=False를 사용
- n_jobs 매개변수는 사이킷런이 훈련과 예측에 사용할 CPU 코어 수를 지정한다.

앙상블은 비슷한 편향에서 더 작은 분산을 만든다.(오차수는 비슷하지만 결정 경계는 덜 불규칙)

부트스트래핑은 각 예측기가 학습하는 서브셋에 다양성을 증가시키므로 배깅이 페이스팅보다 편향이 조금 더 높다.

### oob  평가

배깅을 사용하면 어떤 샘플은 한 예측기를 위해 여러 번 샘플링되고 어떤 것은 전혀 선택되지 않는다.

BaggingClassifier 는 기본값으로 중복을 허용하여 훈련 세트의 크기만큼인 m 개 샘플을 선택한다.
> 평균적으로 63% 정도만 샘플링이 된다.
> 선택되지 않은 훈련 샘플의 나머지 37%를 oob(out-of-bag) 샘플이라고 한다.

예측기가 훈련되는 동안 oob 샘플은 사용되지 않으므로 별도의 검증 세트를 사용하지 않고 oob 샘플을 사용해 평가할 수 있다.

### 랜덤 패치와 랜덤 서브스페이스

**BaggingClassifier**는 특성 샘플링도 지원한다.
- 샘플링은 **max_features, bootstrap_features** 두 매개변수로 조절
- 샘플이 아닌 특성에 대한 샘플링
- 각 예측기는 무작위로 선택한 입력 특성의 일부분으로 훈련된다.
- 매우 고차원의 데이터셋을 다룰 때 유용하다

**랜덤 패치 방식(random patches method)**
- 훈련 특성과 샘플을 모두 샘플링하는 방식

**랜덤 서브스페이스 방식(random subspaces method)**
- 훈련 샘플을 모두 사용하고 특성은 샘플링하는 방식


### 랜덤 포레스트

- 일반적으로 배깅 방법을 적용한 결정 트리의 앙상블
- max_sample : 훈련 세트 크기 지정
- **BaggingClassifier**에 **DecisionTreeClassifier**를 넣어 만든다.
- 결정 트리에 최적화 되어 사용하기 편리한 **RandomForestClassifier**를 사용할 수 있다.

**RandomForestClassifier**는 몇 가지 예외가 있지만 **Decision Tree Classifier**의 매개변수와 앙상블 자체를 제어하는 데 필요한 **BaggingClassifier**의 매개변수를 모두 가지고 있다.

랜덤 포레스트 알고리즘은 트리의 노드를 분할할 때 전체 특성 중에서 최선의 특성을 찾는 대신 무작위로 선택한 특성 후보 중에서 최적의 특성을 찾는 식으로 무작위성을 더 주입한다.


<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA0MDE5Mzk2MiwtMjAwNTMzNDU3MCwtMj
A1NTA1ODQ1NCwyMDQ4MDU2MzYzLDE2NjgwOTkxOTgsLTEwOTM1
MDg5LC0xODU1NjM2NDg0LDE3MjEwMzk2ODMsLTE3NTc0NzIzNz
YsLTExNTMzOTA0ODQsLTEwNDA0NDM4MjQsLTE5MzYxMTgsNTk5
NTkyODcyLDYzNDQxOTU2MCwxODU0MTc4MDk0LDk2Mzc2MTQ0My
wtNjU2ODcxNzg2LC0xOTUzODQzMDQ4LC0xMzg1MTUzMzAyLDE3
MzM2NDY5OTJdfQ==
-->