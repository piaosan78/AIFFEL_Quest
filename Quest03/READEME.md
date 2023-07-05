
# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 박영준
- 리뷰어 : 김소연

# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 각 코드마다 주석이 잘 작성되어 있어서 실행과정이 잘 이해되었습니다.
- [O] 코드가 에러를 유발할 가능성이 없나요?
  >마지막 counter 클래스를 사용해서 전체 2-gram의 빈도수를 출력하는 부분 외에 텍스트 전처리, 2gram생성, 빈도 계산 등 단계가 명확하게 구분되어 있습니다.
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 코드의 작성과정을 잘 설명해주셨습다.
- [O] 코드가 간결한가요?
  > counter클래스를 사용해서 빈도를 계산하고 가장 빈도가 높은 2gram찾는 등 기능이 간결하게 구현되어 있습니다.

# 예시
```python
from collections import Counter
from google.colab import drive                 #코랩을 통해 구글 드라이브 활용
import re
#1. 스크립트 파일 읽기
drive.mount('/content/drive')

with open('/content/06TheAvengers.txt') as f:
    script = f.read()

#2. 스크립트 파일내 텍스트 소문자화로 재정의
script = script.lower()

#3. 스크립트 파일 기호 제거
#symbols = ['.', ',', '!', '?', "'", '"', '~', '-', "\" ]
#for symbol in symbols:
#    script = script.replace(symbol, '')

script = re.sub('[^a-zA-Z\s]', '', script)

#4. 스크립트 파일 단어별 분류
words = script.split()

#5. 스크립트 파일 단어 기준 2-gram 생성
ngrams = zip(words, words[1:])

#6. 2-gram 빈도 계산
countdict = Counter(ngrams)

#7. 가장 빈도가 높은 2-gram 찾기
max2gram = max(countdict, key=countdict.get)

#8. 결과 출력
print(max2gram, countdict[max2gram])
print(Counter(countdict))
```

# 참고 링크 및 코드 개선
```python

```
