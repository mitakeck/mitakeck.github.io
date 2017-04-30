+++
Categories = []
Description = ""
Tags = ["python", "Azure"]
date = "2017-04-30T16:40:01+09:00"
title = "Text-to-Speech API 使ってみる"

+++

# Text-to-Speech API 使ってみる

- 使用するライブラリ群

```python
import http.client, urllib.parse, json
```

- アクセストークンを取得する

```python
subscription_key = "< サブスクリプションキー >"
headers = {
    "Content-type":              "application/x-www-form-urlencoded",
    "Ocp-Apim-Subscription-Key": subscription_key
}
access_token_host = "api.cognitive.microsoft.com"
path = "/sts/v1.0/issueToken"

conn = http.client.HTTPSConnection(access_token_host)
conn.request("POST", path, params, headers)
response = conn.getresponse()

print(response.status, response.reason)

data = response.read()
conn.close()

access_token = data.decode("UTF-8")
print ("Access Token: " + access_token)
```

- 音声データを作成する

```python
text = "日本語しゃべります"
body = "<speak version='1.0' xml:lang='en-us'><voice xml:lang='ja-jp' xml:gender='Female' name='Microsoft Server Speech Text to Speech Voice (ja-JP, Ayumi, Apollo)'>" + text + "</voice></speak>"

headers = {
    "Content-type":             "application/ssml+xml",
    "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
    "Authorization":            "Bearer " + access_token,
    "User-Agent":               "Meriken"
}

# https://speech.platform.bing.com/synthesize
conn = http.client.HTTPSConnection("speech.platform.bing.com")
conn.request("POST", "/synthesize", body.encode('utf-8'), headers)
response = conn.getresponse()
print(response.status, response.reason)

data = response.read()
conn.close()

wav = open("download-" + text + ".wav", "wb")
wav.write(data)
wav.close()
```

声色は何種類か用意されていて、日本語は男性声と女性声と 2 種類ある

```
ar-EG*	Female	"Microsoft Server Speech Text to Speech Voice (ar-EG, Hoda)"
de-DE	Female	"Microsoft Server Speech Text to Speech Voice (de-DE, Hedda)"
de-DE	Male	"Microsoft Server Speech Text to Speech Voice (de-DE, Stefan, Apollo)"
en-AU	Female	"Microsoft Server Speech Text to Speech Voice (en-AU, Catherine)"
en-CA	Female	"Microsoft Server Speech Text to Speech Voice (en-CA, Linda)"
en-GB	Female	"Microsoft Server Speech Text to Speech Voice (en-GB, Susan, Apollo)"
en-GB	Male	"Microsoft Server Speech Text to Speech Voice (en-GB, George, Apollo)"
en-IN	Male	"Microsoft Server Speech Text to Speech Voice (en-IN, Ravi, Apollo)"
en-US	Female	"Microsoft Server Speech Text to Speech Voice (en-US, ZiraRUS)"
en-US	Male	"Microsoft Server Speech Text to Speech Voice (en-US, BenjaminRUS)"
es-ES	Female	"Microsoft Server Speech Text to Speech Voice (es-ES, Laura, Apollo)"
es-ES	Male	"Microsoft Server Speech Text to Speech Voice (es-ES, Pablo, Apollo)"
es-MX	Male	"Microsoft Server Speech Text to Speech Voice (es-MX, Raul, Apollo)"
fr-CA	Female	"Microsoft Server Speech Text to Speech Voice (fr-CA, Caroline)"
fr-FR	Female	"Microsoft Server Speech Text to Speech Voice (fr-FR, Julie, Apollo)"
fr-FR	Male	"Microsoft Server Speech Text to Speech Voice (fr-FR, Paul, Apollo)"
hi-IN	Female	"Microsoft Server Speech Text to Speech Voice (hi-IN, Kalpana, Apollo)"
it-IT	Male	"Microsoft Server Speech Text to Speech Voice (it-IT, Cosimo, Apollo)"
ja-JP	Female	"Microsoft Server Speech Text to Speech Voice (ja-JP, Ayumi, Apollo)"
ja-JP	Male	"Microsoft Server Speech Text to Speech Voice (ja-JP, Ichiro, Apollo)"
ko-KR	Female	"Microsoft Server Speech Text to Speech Voice (ko-KR,HeamiRUS)"
pt-BR	Male	"Microsoft Server Speech Text to Speech Voice (pt-BR, Daniel, Apollo)"
ru-RU	Female	"Microsoft Server Speech Text to Speech Voice (ru-RU, Irina, Apollo)"
ru-RU	Male	"Microsoft Server Speech Text to Speech Voice (ru-RU, Pavel, Apollo)"
zh-CN	Female	"Microsoft Server Speech Text to Speech Voice (zh-CN, HuihuiRUS)"
zh-CN	Female	"Microsoft Server Speech Text to Speech Voice (zh-CN, Yaoyao, Apollo)"
zh-CN	Male	"Microsoft Server Speech Text to Speech Voice (zh-CN, Kangkang, Apollo)"
zh-HK	Female	"Microsoft Server Speech Text to Speech Voice (zh-HK, Tracy, Apollo)"
zh-HK	Male	"Microsoft Server Speech Text to Speech Voice (zh-HK, Danny, Apollo)"
zh-TW	Female	"Microsoft Server Speech Text to Speech Voice (zh-TW, Yating, Apollo)"
zh-TW	Male	"Microsoft Server Speech Text to Speech Voice (zh-TW, Zhiwei, Apollo)"
```

## 参照リンク

- https://docs.microsoft.com/ja-jp/azure/cognitive-services/Speech/api-reference-rest/bingvoiceoutput
- https://github.com/Azure-Samples/Cognitive-Speech-STT-JavaScript
- https://gist.github.com/mitakeck/b2a953dfdf84b1e2c0d3c950d812ad0f
