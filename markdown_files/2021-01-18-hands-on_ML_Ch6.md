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


<!--stackedit_data:
eyJoaXN0b3J5IjpbMTYyMTI5OTgzNywtMTE1MTg3NzE0OCwyMD
I5MDc2NTI2LC01MzY0NDAwMzUsLTIwODg3NDY2MTJdfQ==
-->