# ObjectDetectionPractice

下記コマンド実行すると`picamera`のインストールでエラーが発生する
```shell
$ pipenv install
```
コマンド実行後、次のコマンドも実行する
```shell
$ pipenv install picamera
```

## 実行
入力画像は`input_images`内に配置する.
```shell
$ pipenv run detect_image --image input_images/[画像ファイル名]
```
