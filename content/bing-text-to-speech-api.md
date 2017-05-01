+++
Categories = []
Description = ""
Tags = ["python", "Azure"]
date = "2017-04-30T16:40:01+09:00"
title = "Azure の Text-to-Speech API 使ってみる"

+++

# Azure のText-to-Speech API 使ってみる

## 概要

Text-to_Speech とは文字データから合成発声データを作成する技術のことである

基本的には [Microsoft の Cognitive Services の API リファレンス](https://docs.microsoft.com/ja-jp/azure/cognitive-services/Speech/api-reference-rest/bingvoiceoutput) に従って API を叩けばできるが、音声データ作成処理に手間取ったりしたのでメモ書きとして残しておく

Cognitive Service を Python から叩くことを想定している

一連の流れをまとめたファイルは以下から閲覧できる
https://gist.github.com/mitakeck/b2a953dfdf84b1e2c0d3c950d812ad0f

## アクセストークンを取得する

なんか知らないけど、Cognitive Searvice のサブスクリプションキーを Azure で発行した後に、アクセストークンを取得しないといけないらしい

Header に `Ocp-Apim-Subscription-Key` をキーとして、サブスクリプションキーを挿入して `https://api.cognitive.microsoft.com/sts/v1.0/issueToken` へ　POST すると返却値にアクセストークンが渡ってくる

ここで取得したアクセストークンは音声データを作成する際に使いまわすものになる

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

## 音声データを作成する

音声データを作成するには `https://speech.platform.bing.com/synthesize` にデータを投げれば良い

上記のエンドポイントに対して喋らせる内容や声色はリクエストの POST クエリパラメータに [SSML 形式](https://www.w3.org/TR/speech-synthesis/) で入れて、アクセストークンや音声ファイルの形式なんかはリクエストの Header に入れればおｋ

プログラム中では日本語のテキストデータを与えると、発声データが wav ファイルとしてレスポンスデータに乗っかってくるので、ファイルとして書き出して保存する処理をしている

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

実際に作成された wav ファイルがこれ

<audio controls="" width="250px" height="150px">
    <!-- <source src="sample.mp3" type="audio/mp3"> -->
    <source src="sound1.wav" type="audio/wav">
    <embed src="sound1.wav" type="audio/wav" width="240" height="50" autostart="false" controller="true" loop="false" pluginspage="http://www.apple.com/jp/quicktime/download/">
</audio>

## 声色を変えてみる

声色は何種類か用意されていて、日本語は男性声と女性声と 2 種類ある

```
ar-EG*	Female
de-DE	Female
de-DE	Male
en-AU	Female
en-CA	Female
en-GB	Female
en-GB	Male
en-IN	Male
en-US	Female
en-US	Male
es-ES	Female
es-ES	Male
es-MX	Male
fr-CA	Female
fr-FR	Female
fr-FR	Male
hi-IN	Female
it-IT	Male
ja-JP	Female
ja-JP	Male
ko-KR	Female
pt-BR	Male
ru-RU	Female
ru-RU	Male
zh-CN	Female
zh-CN	Female
zh-CN	Male
zh-HK	Female
zh-HK	Male
zh-TW	Female
zh-TW	Male
```

英語の男性声でちょっと長めの文章を喋らせてみる

```
text = "The Voice Browser Working Group has sought to develop standards to enable access to the Web using spoken interaction. The Speech Synthesis Markup Language Specification is one of these standards and is designed to provide a rich, XML-based markup language for assisting the generation of synthetic speech in Web and other applications. The essential role of the markup language is to provide authors of synthesizable content a standard way to control aspects of speech such as pronunciation, volume, pitch, rate, etc. across different synthesis-capable platforms."
body = "<speak version='1.0' xml:lang='en-us'><voice xml:lang='en-US' xml:gender='Female' name='Microsoft Server Speech Text to Speech Voice (en-US, BenjaminRUS)'>" + text + "</voice></speak>"

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

wav = open("sound2.wav", "wb")
wav.write(data)
wav.close()
```

出力結果はこれ

<audio controls="" width="250px" height="150px">
    <!-- <source src="sample.mp3" type="audio/mp3"> -->
    <source src="sound2.wav" type="audio/wav">
    <embed src="sound2.wav" type="audio/wav" width="240" height="50" autostart="false" controller="true" loop="false" pluginspage="http://www.apple.com/jp/quicktime/download/">
</audio>

いい感じ

## 参照リンク

- https://docs.microsoft.com/ja-jp/azure/cognitive-services/Speech/api-reference-rest/bingvoiceoutput
- https://github.com/Azure-Samples/Cognitive-Speech-STT-JavaScript
- https://gist.github.com/mitakeck/b2a953dfdf84b1e2c0d3c950d812ad0f
