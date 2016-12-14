# Dependencies
 - Theano (or Tensorflow)
 - Keras
 - spotipy
 - discogs_client

# 抓取專輯資料

首先，執行 `run addnew.py`：
## 查詢專輯
使用 `getAlbumDetail(${專輯名稱})`

    getAlbumDetail('Wild Light')

    > {u'album': u'Wild Light',
    >  u'artist': u'65daysofstatic',
    >  u'coversize': True,
    >  u'coverurl':
    >  u'https://i.scdn.co/image/fe8ed972cf16049d6157985c6e1dc990b8f7b9ca',
    >  u'genres': [u'electronic',u'rock',u'postrock', u'mathrock', u'shoegaze'],
    >  u'release_date': u'2013-10-29'}
因為圖片格式目前只能固定在 `300x300`，所以 `coversize` 為 `True` 才有辦法儲存。

## 新增專輯
使用 `addAlbum(${專輯名稱}, ${目錄路徑})`

    addAlbum('Wild Light', 'data/set2/')
如果沒錯誤訊息，`data/set2/albumlabel.csv` 會多增加一行資料，`data/set2/img` 裡也會多一張圖。
