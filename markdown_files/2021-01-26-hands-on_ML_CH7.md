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

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTM1NDY1MzE1Nyw5Mjc0MDgwNjgsLTEyMj
EyMjc4MDAsLTExOTI5OTU3MTldfQ==
-->