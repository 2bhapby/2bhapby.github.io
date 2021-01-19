# 결정 트리

- 분류, 회귀, 다중출력 작업이 가능한 알고리즘
- 매우 복잡한 데이터셋도 학습 가능

## 결정 트리 학습과 시각화

![iris decision tree](https://2bhapby.github.io/images/iris_decision_tree.PNG)

graphviz를 사용하면 위와 같이 결정트리를 시각화 시킬 수 있다.

## 예측하기

samples : 얼마나 많은 훈련 샘플이 적용되었는가
value : 노드에서 각 클래스에 얼마나 많은 훈련 샘플이 있는지
gini : 불순도(impurity) 측정
> 한 노드의 모든 샘플이 같은 클래스에 속해 있다면 이 노드를 순수하다고 한다.
> $G_i = 1 - \sum_{k=1}^{n}p_{i, k}^3$
> 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIxMTMwOTgzNDksLTUzNjQ0MDAzNSwtMj
A4ODc0NjYxMl19
-->