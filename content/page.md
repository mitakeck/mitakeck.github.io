+++
title = "hugo test"
date = "2017-03-06T00:38:19+09:00"
+++

# hugo のテスト


- シンタックスハイライトとか

```zsh
% curl -Lso-  http://mainichi.jp/articles/20151210/k00/00m/040/010000c | grep meta |  grep og:image | sed -e "s/.*content=\"\(.*\)\".*/\1/g"
# => http://cdn.mainichi.jp/vol1/2015/12/10/20151210k0000m040023000p/91.jpg

% curl -Lso-  http://mainichi.jp/articles/20151210/k00/00m/040/010000c | grep meta |  grep og:image | sed -e "s/.*content=\"\(.*\)\".*/\1/g"
# => http://cdn.mainichi.jp/vol1/2015/12/10/20151210k0000m040023000p/91.jpg

% curl -Lso-  http://www.asahi.com/articles/ASHD96K3YHD9UTIL04L.html | grep meta |  grep og:image | sed -e "s/.*content=\"\(.*\)\".*/\1/g"
# => http://www.asahicom.jp/articles/images/AS20151209004367_comm.jpg

% curl -Lso- http://www.cnn.co.jp/world/35074672.html  | grep meta |  grep og:image | sed -e "s/.*content=\"\(.*\)\".*/\1/g"              
# => http://www.cnn.co.jp/storage/2015/12/09/26290e751df22d92b3d72b36a65d95b3/france-russia-police-dog.jpg

% curl -Lso-  http://ggsoku.com/2015/12/minecraft-for-wii-u-released-1217/ | grep meta |  grep og:image | sed -e "s/.*content=\"\(.*\)\".*/\1/g"
# => http://ggsoku.com/index.php?aam_media=101921&#038;size=original

%
```
