# Dependencies
 - Theano (or Tensorflow)
 - Keras
 - spotipy
 - discogs_client

# `dl.py`
概念上只有 `CNN` + 兩層 `Full-Connected`。
輸出為各個 label 的 sigmoid activation，loss 為 binary cross-entropy --- [參考來源](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
# 抓取專輯資料

首先，執行 `run addnew.py`：
## 查詢專輯
使用 `getAlbumDetail(${專輯名稱})`

    getAlbumDetail('Wild Light')

    > u'coverurl': u'https://api-img.discogs.com/BGI98JDQsOh5YOauWy69Vt-8DeA=/fit-in/150x150/filters:strip_icc():format(jpeg):mode_rgb():quality(40)/discogs-images/R-4910396-1379177088-6115.jpeg.jpg',
    > u'genres': [u'electronic', u'rock', u'postrock', u'mathrock', u'shoegaze'],
    > u'release_date': u'2013',
    > u'title': u'65daysofstatic - Wild Light'

## 新增專輯
使用 `addAlbum(${專輯名稱}, ${目錄路徑})`

    addAlbum('Wild Light', 'data/set2/')
如果沒錯誤訊息，`data/set2/albumlabel.csv` 會多增加一行資料，`data/set2/img` 裡也會多一張圖。

## 新增廠牌的全部專輯
首先要找到廠牌在 Discogs 的 Label ID，我想在網頁上 URL 最後那邊都找得到。

    setpath = 'data/set4/'
    label_id = 31269

    import discogs_client
    d = discogs_client.Client('ExampleApplication/0.1', user_token="tPgLfOQMObTxlKYXoSKpZkbVODRLZFqSwPzngIrb")

    l = d.label(label_id)
    lst = [ release.id for release in l.releases]

    for i in xrange(len(lst)):
        try:
            addAlbumByID(lst[i], setpath)
            print("Progress: {}/{} success".format(i+1, len(lst)))
        except:
            print("Progress: {}/{} error".format(i+1, len(lst)))
## 壓縮
因為使用了 `Discogs API` 所提供的圖片，並沒有限制其圖片大小，所以在 `set${x}` 已經收集夠多的圖片後，需要修改 `resize.py`:

    setpath='data/set${x}/'

再執行它。
