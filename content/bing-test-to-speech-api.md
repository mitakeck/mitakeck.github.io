+++
Categories = []
Description = ""
Tags = ["python", "Azure"]
date = "2017-04-30T16:40:01+09:00"
title = "Text-to-Speech API 使ってみる"

+++

# Text-to-Speech API 使ってみる

```python
import http.client, urllib.parse, json
```

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

## 参照リンク

- https://docs.microsoft.com/ja-jp/azure/cognitive-services/Speech/api-reference-rest/bingvoiceoutput
- https://github.com/Azure-Samples/Cognitive-Speech-STT-JavaScript
- https://gist.github.com/mitakeck/b2a953dfdf84b1e2c0d3c950d812ad0f
