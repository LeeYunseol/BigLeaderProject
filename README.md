# BigLeaderProject

데이터 청년 캠퍼스 빅리더AI아카데미

Data Campus Big Leader AI Academy

신장과 심장암 Segmentation 

Segmentation between kidney and cancer

홈페이지는 Flask를 통해서 구현하였습니다.

Home page is implemented through Flask

Unet을 2번 사용하여 처음에는 신장과 신장암 / 배경으로 구분하고 신장과 신장암을 구분한 것에서 다시 신장과 신장암으로 Segmentation을 진행해서 정확도를 높이고자 하였습니다.

We tried to increase accuracy by first dividing (the kidney and kidney cancer) and background. Then, kidney and kidney cancer by using Unet twice.

저는 Unet을 활용하여 Image Segmentation과 백엔드 부분을 구현하였습니다.

I implemented the image segmentation using Unet and backend with Flask.

캐글 대회까지 참여 (16/27) https://www.kaggle.com/competitions/body-morphometry-kidney-and-tumor
