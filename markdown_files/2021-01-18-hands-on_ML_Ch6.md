# 결정 트리

- 분류, 회귀, 다중출력 작업이 가능한 알고리즘
- 매우 복잡한 데이터셋도 학습 가능

## 결정 트리 학습과 시각화

![iris decision tree](https://2bhapby.github.io/images/iris_decision_tree.PNG)

graphviz를 사용하면 위와 같이 결정트리를 시각화 시킬 수 있다.

## 예측하기

**samples**
- 얼마나 많은 훈련 샘플이 적용되었는가

**value**
- 노드에서 각 클래스에 얼마나 많은 훈련 샘플이 있는지

**gini**
- 불순도(impurity) 측정
> 한 노드의 모든 샘플이 같은 클래스에 속해 있다면 이 노드를 순수하다고 한다.
> $G_i = 1 - \sum_{k=1}^{n}p_{i, k}^3$

**화이트 박스**
- 직관적이고 결정 방식을 이해하기 쉽다.(결정 트리)

**블랙 박스**
- 성능이 뛰어나고 예측을 마드는 연산 과정을 쉽게 확인할 수 있다
- 왜 그런 예측을 만드는지는 쉽게 설명하기 어렵다.

## 클래스 확률 추정

샘플에 대한 리프 노드를 찾기 위해 트리를 탐색하고 그 노드에 있는 클래스 $k$의 훈련 샘플의 비율을 반환한다.

## CART 훈련 알고리즘

> CART 알고리즘은 greedy 이다.
> 최적의 트리를 찾는것은 NPC 문제이다.

결정 트리를 훈련시키기 위해 CART(classification and regression tree) 알고리즘을 사용한다.

- 분류에 대한 CART 비용함수
-- $J(k, t_k) = \frac{m_{left}}{m}G_{left} + \frac{m_{right}}{m}G_{right}$
>$G_{left/right}$는 왼쪽/오른쪽 서브셋의 불순도
>$m_{left/right}$는 왼쪽/오른쪽 서브셋의 샘플 수

서브셋을 나누면서 최대 깊이가 되면 중지하거나 불순도를 줄이는 분할을 찾을 수 없을 때 멈추게 된다.
- 중지 조건
--min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes)

## 계산복잡도

평균적으로 $O(log_2(m))$ 이다.

훈련 세트가 적을 경우 사이킷런은 **presort=True** 매개변수를 통해 미리 데이터를 정렬하여 훈련 속도를 높일 수 있다.

훈련세트가 많다면 속도가 많이 느려진다.

## 지니 불순도 또는 엔트로피?

기본적으로는 **지니 불순도**를 사용한다. 
하지만 criterion 매개변수를 "entropy"로 지정하여 **엔트로피 불순도**를 사용할 수 있다.

**$i$ 번째 노드의 엔트로피**
- $H_i = - \sum_{\overset{k = 1}{p_{i, k \neq 0}}}^{n}p_{i, k}log_2(p_{i,k})$
-- 어떤 세트에 한 클래스의 샘플만 담고 있다면 엔트로피는 0

**계산 속도**
- 지니 불순도 > 엔트로피
> 지니 불순도를 기본값으로 하는게 좋음
> 
**균형 잡힌 트리**
- 엔트로피 > 지니 불순도
>지니 불순도가 가장 빈도 높은 클래스를 한쪽 가지로 고림시키는 경향이 있다.

## 규제 매개변수

**선형 모델**은 데이터가 선형이라고 가정하지만 **결정 트리**는 훈련 데이터에 대한 제약이 거의 없다.

하지만 제약을 두지 않으면 훈련데이터에 맞추려해 **overfitting** 되기 쉽다.

---
결정 트리는 **비파라미터(nonparametric model) 모델**이다.
- 훈련되기 전에 파라미터 수가 결정되지 않는 모델
- 자유도가 높고 overfitting 되기 쉽다.

선형 모델은 **파라미터 모델(parametric model)** 이다.
- 미리 정의된 모델 파라미터 수를 가진다.
- 자유도가 제한되고 과대적합 위험이 줄어든다.

과대 적합을 피하기 위해 결정 트리의 **자유도를 제한**할 필요가 있다.

- 사이킷런에서는 max_depth 매개변수로 결정 트리의 **최대 깊이**를 제어한다.

**DecisionTreeClassifier** 에는 비슷하게 결정 트리의 형태를 제한하는 다른 매개변수가 몇개 있다.
- min_samples_split
--분할되기 위해 노드가 가져야 하는 최소 샘플 수
- min_samples_leaf
-- 리프 노드가 가지고 있어야 할 최소 샘플 수
- min_weight_fraction_leaf
-- min_samples_leaf와 같지만 가중치가 부여된 전체 샘플 수에서의 비율
- max_leaf_nodes
-- 리프 노드의 최대 수
- max_features
-- 각 노드에서 분할에 사용할 특성의 최대 수


 min_으로 시작하는 매개변수를 증가시키거나 max_로 시작하는 매개변수를 감소시키면 규제가 커진다.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2MTc1NzAyMTAsMzY1ODUxNzAxLDE3OD
MyMjY5MTYsLTExNTE4NzcxNDgsMjAyOTA3NjUyNiwtNTM2NDQw
MDM1LC0yMDg4NzQ2NjEyXX0=
-->