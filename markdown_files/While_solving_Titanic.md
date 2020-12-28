---
layout:post
title:titanic 문제를 풀며
---

# Titanic 문제를 풀며  
  

Titanic 문제를 풀면서 느낀 점은 아직은 어떠한 알고리즘을 써야할지 모른다는 것이다. 분류 알고리즘에는 여러 종류가 있다. 하지만 이 알고리즘의 장단점이 무엇인지 어떤 문제에서 유리한지에 대한 기본적인 지식도 없다는 문제가 있다는 것을 깨달았다.  
  

물론 단순히 black-box 처럼 모든 classifier을 가져다 쓸 수도 있지만 이는 좋은 방법이 아니라고 생각하기에 여러 classifier와 그의 장단점에 대해서 찾아보려고 한다  
  
  

## Classifier의 종류  
  

 1. kNN  
 2. Decision Tree  
 3. Random Forest  
 4. SVM  
  

### kNN (k - Nearest Neighborhood)  
  

 - 간단하게 "친구를 보면 그 사람을 알 수 있다" 라는 말로 설명할 수 있다.  
 - 즉, 자신과 가장 가까운 k 개의 데이터를 보고 그 데이터가 무엇인지 예측하는 것이다.  
 - 문제점은 적절한 k 값을 구하는 것이다.  
  

### Decision Tree  
  

- Decision Tree를 통해서 데이터에 있는 규칙을 학습하고 예측하는 classifier이다.  
-  주의해야 할 점은 규칙이 많아질수록 overfitting이 된다는 것이다.  
-    
  

### Random Forest  
  

- Decision Tree가 여러개 모여서 이룬것이다.  
- 각 Tree들이 낸 결론 중 더 많은 예측을 따른다.  
  
  
  
  

### SVM  

- Decision Boundary(분류 기준 선)를 통해서 예측하는 것이다.  
- 장점으로는 2차원이 아닌 다중차원에서 분류하는데 특화되어 있고 사용이 쉽다는 점이 있다.  
- 단점으로는 모델 파라미터를 조절하기 위해서 여러번의 테스트가 필요하고 결과에 대한 설명력이 떨어진다.  
    
  

reference:  
https://nittaku.tistory.com/286  
https://gomguard.tistory.com/51  
https://injo.tistory.com/15#Decision-Tree(%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC)-:  
https://nonmeyet.tistory.com/entry/R-SVM-%EC%84%9C%ED%8F%AC%ED%8A%B8-%EB%B2%A1%ED%84%B0-%EB%A8%B8%EC%8B%A0%EC%9D%98-%EC%A0%95%EC%9D%98%EC%99%80-%EC%84%A4%EB%AA%85 
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDc2MDE2MDM0LC0yNjIwNTUzNjJdfQ==
-->